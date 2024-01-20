import dataclasses
from typing import List
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os
from logging import Logger
import logging
import faiss
import torch
from datetime import datetime as dt
import wandb
import tqdm
import glob
import shutil

import json
import gc
import sys

@dataclasses.dataclass
class Config:
    extract_file_name: str  # 作成した embedding を保存するファイル名
    experiment_name: str # 実験名
    wiki_file_names: List[str]
    target_texts_name: str = "data/gpt_generated_data/only_a_250text.csv"

    # feature extractor
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    feature_extractor: str = "sentence_transformer"
    batch_size: int = 128
    pretrained_model_path: str = None
    pretrain_index_dir: str = ""
    
    # text preprocess
    wiki_text_preprocess: str = "all"
    target_text_preprocess: str = "only_prompt"
    debug: bool = False
    max_length: int = 256
    
    # faiss index
    faiss_index: faiss.Index = faiss.IndexIVFPQ
    faiss_index_parameter: dict = dataclasses.field(
        default_factory=lambda: {"nlists": 8, "M": 64, "nbits": 8}
    )
    
    # calc_mapk
    calculation_map_recall: bool = True

def get_logger(output_dir: str):
    """
    logger を作成する. formatter は "%Y-%m-%d %H:%M:%S" で作成する.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # formatter
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    
    # handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    handler = logging.FileHandler(f"{output_dir}/log.txt", "w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

class FeatureExtractor:
    def __init__(
        self,
        logger: Logger,
        config: Config,
    ):
        self.config = config
        self.logger = logger
        self.text_preprocess = self._get_text_preprocess()

    def _get_text_preprocess(self):
        raise NotImplementedError
    
    def _extract_feature_sentence_transformer(
        self,
        texts: List[str],
    ):
        model = SentenceTransformer(self.config.model_name)
        if self.config.max_length is not None:
            model.max_seq_length = self.config.max_length
        
        # 1万件ずつに分割して計算する
        embeddings = []
        for i in tqdm.tqdm(range(0, len(texts), 10000), desc="extract feature"):
            embeddings.append(
                model.encode(texts[i:i+10000], batch_size=self.config.batch_size, show_progress_bar=False)
            )
        
        embeddings = np.concatenate(embeddings)
        return embeddings
    
    def _extract_feature_custom_bert(
        self,
        texts: List[str],
    ):
        with open(f"{os.path.dirname(self.config.pretrained_model_path)}/config.json", "r") as f:
            config_retrieval = json.load(f)
            config_retrieval = RetrievalConfig(**config_retrieval)
        
        model = RetrievalNet(
            cfg=config_retrieval,
            logger=self.logger,
        )
        self.logger.info(f"load pretrained weight from {self.config.pretrained_model_path}")
        model.load_state_dict(torch.load(self.config.pretrained_model_path))
        model.to("cuda")
        
        embeddings = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for i in tqdm.tqdm(range(0, len(texts), 10000), desc="extract feature"):
                df = pd.DataFrame({"text": texts[i:i+10000], "length": [len(text) for text in texts[i:i+10000]]})
                dataset = RetrievalDataset(df, config_retrieval)
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=config_retrieval.batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                )
                for batch in loader:
                    text = batch[0].to("cuda")
                    attention_mask = batch[1].to("cuda")
                    output = model.forward(text, attention_mask)
                    embeddings.append(output.cpu().detach().numpy())
        embeddings = np.concatenate(embeddings)
        return embeddings
    
    def _extract_feature_bert(
        self,
        texts: List[str],
    ):
        model = pipeline("feature-extraction", model=self.config.model_name, torch_dtype=torch.float16)
        model.tokenizer.model_max_length = self.config.max_length

        embeddings = []
        for i in tqdm.tqdm(range(0, len(texts), 10000), desc="extract feature"):
            embeddings.append(
                model(texts[i:i+10000], batch_size=self.config.batch_size)
            )
        embeddings = np.concatenate(embeddings)
        return embeddings

    def extract_feature(
        self, 
        df: pd.DataFrame,
        file: str = "",
    ):
        """
        与えられたテキストから embedding を作成する. 
        """
        
        self.logger.info("extract feature start")
        texts = self._preprocess_texts(df).values.tolist()
        # すでに embedding が作成されている場合は, それを返す
        extract_fname = f"{self.config.extract_file_name}_{os.path.basename(file).split('.')[0]}.npy"
        if os.path.isfile(extract_fname) and self.__class__.__name__ == "FeatureExtractorFromWiki":
            self.logger.info(f"すでに embedding が作成されているため, それを返します: {extract_fname}")
            return np.load(extract_fname)

        if self.config.feature_extractor == "custom_bert":
            embeddings = self._extract_feature_custom_bert(texts)
        elif self.config.feature_extractor == "sentence_transformer":
            embeddings = self._extract_feature_sentence_transformer(texts)
        elif self.config.feature_extractor == "bert":
            embeddings = self._extract_feature_bert(texts)
        else:
            raise ValueError()
        if self.__class__.__name__ == "FeatureExtractorFromWiki":
            self.logger.info(f"embedding を保存します: {extract_fname}")
            np.save(extract_fname, embeddings)

        return embeddings
        

class FeatureExtactorFromPrompt(FeatureExtractor):
    """
    与えられた prompt から embedding を作成する. 
    """
    
    def _get_text_preprocess(self):
        return self.config.target_text_preprocess

    def _preprocess_texts(
        self, 
        df: pd.DataFrame,
    ):
        if self.text_preprocess == "only_prompt":
            return df["prompt"].astype(str)
        if self.text_preprocess == "prompt_and_choice":
            return (
                df["prompt"].astype(str) + " [SEP] " + 
                df["A"].astype(str) + " [SEP] " + 
                df["B"].astype(str) + " [SEP] " + 
                df["C"].astype(str) + " [SEP] " + 
                df["D"].astype(str) + " [SEP] " + 
                df["E"].astype(str)
            )
        if self.text_preprocess == "prompt_and_choice_without_sep":
            return (
                df["prompt"].astype(str) + " \n" + 
                df["A"].astype(str) + " \n" + 
                df["B"].astype(str) + " \n" + 
                df["C"].astype(str) + " \n" + 
                df["D"].astype(str) + " \n" + 
                df["E"].astype(str)
            )
        raise NotImplementedError(f"想定していないtarget_text_processです: {self.config.text_preprocess}")

    
    
class FeatureExtractorFromWiki(FeatureExtractor):
    """
    与えられた wiki の text から embedding を作成する. 
    """    
    def _get_text_preprocess(self):
        return self.config.wiki_text_preprocess
    
    def _preprocess_texts(
        self, 
        df: pd.DataFrame,
        title_only: bool = True
    ):
        def f(text):
            return text.split("==")[0]
        
        def edit_text(series):
            if series["title_only"] and title_only:
                return series["title"]
            else:
                return "#" + series["title"] + "\n" + series["text"]
        
        return df.apply(edit_text, axis=1)
        
class IndexCreator:
    def __init__(
        self,
        logger: Logger,
        config: Config,
    ):
        self.config = config
        self.logger = logger
    
    def create_index(
        self,
        embeddings: np.ndarray,
    ):
        """
        与えられた embedding から index を作成する. 
        """
        self.logger.info("create index start")
        embeddings = embeddings.astype(np.float32)
        quantizer = faiss.IndexFlatIP(embeddings.shape[1])
        if self.config.faiss_index == faiss.IndexIVFPQ:
            index = self.config.faiss_index(
                quantizer, 
                embeddings.shape[1], 
                self.config.faiss_index_parameter["nlists"], 
                self.config.faiss_index_parameter["M"], 
                self.config.faiss_index_parameter["nbits"]
            )
        
        print("train")
        index.train(embeddings)
        print("add")
        index.add(embeddings)
        
        return index


def search(index, embeddings, ids, k):
    """
    wiki から作成した embedding と, target から作成した embedding を比較して, 
    類似する prompt を探し, その wiki の id を返す.
    """
    def _search(query, index, k):
        similarity, searched_index = index.search(query, k)
        return similarity, searched_index
    similarities, searched_index = _search(embeddings, index, k=k)
    searched_ids = []
    ret_similarities = []
    for i in tqdm.tqdm(range(len(searched_index))):
        searched_id = []
        similarity = []
        searched_index_ = searched_index[i]
        similarity_ = similarities[i]
        for j, idx in enumerate(searched_index_):
            if idx == -1:
                continue
            searched_id.append(ids[idx])
            similarity.append(similarity_[j])
        searched_ids.append(searched_id)
        ret_similarities.append(similarity)
    return searched_ids, ret_similarities


def mapk(y_true, y_pred, k=10):
    """
    y_true: (n_samples)
    y_pred: (n_samples, n_classes)
    """
    map_score = 0
    for i in range(len(y_true)):
        y_true_ = y_true[i]
        y_pred_ = y_pred[i]
        score = 0
        for j in range(k):
            if y_pred_[j] == y_true_:
                score += 1 / (j + 1)
                break
        map_score += score
    map_score /= len(y_true)
    return map_score

def recallk(y_true, y_pred, k=10):
    """
    y_true: (n_samples)
    y_pred: (n_samples, n_classes)
    """
    map_score = 0
    for i in range(len(y_true)):
        y_true_ = y_true[i]
        y_pred_ = y_pred[i]
        score = 0
        for j in range(min(k, len(y_pred_))):
            if y_pred_[j] == y_true_:
                score += 1
                break
        map_score += score
    map_score /= len(y_true)
    return map_score


def main(config: Config):
    
    try:
        output_dir = f"output/context_pipeline/stage1/{os.path.basename(__file__)}/{dt.now().strftime('%Y%m%d%H%M%S')}_{config.experiment_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        embeddings_wiki = []
        text_ids = []
        wiki_ids = []
        texts_concatenate = []
        if not torch.cuda.is_available():
            raise ValueError("GPUが使えません")
        # wandb の設定
        wandb.init(
            project="llm_science",
            name=config.experiment_name,
            config=config,
            tags=["context_pipeline", "stage1"],
        )
        
        # config を wandb に保存する
        wandb.config.update(config)
        
        logger = get_logger(output_dir=output_dir)
        logger.info("load data")
        shutil.copy(__file__, f"{output_dir}/exp002.py")

        # target_texts から embedding を作成する
        df_target = pd.read_csv(config.target_texts_name).drop_duplicates()

        if "dataset" not in df_target.columns:
            logger.info("dataset(train, val) が定義されていないため、全部 train として扱います")
            df_target["dataset"] = "train"
            
        logger.info(f"train: {len(df_target[df_target['dataset'] == 'train'])}, valid: {len(df_target[df_target['dataset'] == 'valid'])}")

        feature_extractor = FeatureExtactorFromPrompt(config=config, logger=logger)
        # embeddings_target = feature_extractor.extract_feature(df=df_target)
        
        # test data の prompt を探す
        df_test = pd.read_csv("data/kaggle-llm-science-exam/train.csv")
        # embeddings_test = feature_extractor.extract_feature(df=df_test)
        
        df_target["searched_ids"] = [[]] * len(df_target) 
        df_target["searched_text_ids"] = [[]] * len(df_target) 
        df_target["searched_text"] = [[]] * len(df_target) 
        df_target["similarity"] = [[]] * len(df_target) 
        
        df_test["searched_text_ids"] = [[]] * len(df_test) 
        df_test["searched_ids"] = [[]] * len(df_test) 
        df_test["searched_text"] = [[]] * len(df_test) 
        df_test["similarity"] = [[]] * len(df_test) 
        
        text_id_start = 0

        for file in config.wiki_file_names:
            logger.info(f"load file {file}")
            
            if ".parquet" in file:
                df_wiki = pd.read_parquet(file)
            if ".pickle" in file:
                df_wiki = pd.read_pickle(file)
            if ".feather" in file:
                df_wiki = pd.read_feather(file)
                
            if config.debug:
                df_wiki = df_wiki.iloc[:1000]
            
            feature_extractor = FeatureExtractorFromWiki(config=config, logger=logger)
            df_wiki["text_id"] = np.arange(len(df_wiki)) + text_id_start
            text_id_start += len(df_wiki)
            texts_concatenate = feature_extractor._preprocess_texts(df_wiki, title_only=False).values.tolist()
            wiki_ids = df_wiki["id"].values.tolist()
            text_ids = df_wiki["text_id"].values.tolist()
            wiki_dict = dict(zip(text_ids, texts_concatenate))
    
            # wiki から embedding を作成する
            if os.path.isfile(f"{config.pretrain_index_dir}/index_{os.path.basename(file).split('.')[0]}.faiss" ):
                logger.info("index がすでに存在するので読み込みます")
                index = faiss.read_index(f"{config.pretrain_index_dir}/index_{os.path.basename(file).split('.')[0]}.faiss")
            else:
                embeddings_wiki = feature_extractor.extract_feature(df=df_wiki, file=file)
                
                # embedding から index を作成する    
                index_creator = IndexCreator(config=config, logger=logger)
                index = index_creator.create_index(embeddings=embeddings_wiki)
        
                # faiss index を保存する                
                faiss.write_index(index, f"{output_dir}/index_{os.path.basename(file).split('.')[0]}.faiss")
            # 事前に作成した faiss index を使って類似する prompt を探す
            # searched_ids, _ = search(index=index, embeddings=embeddings_target, ids=wiki_ids, k=30)
            # df_target["searched_ids"] = [x + searched_ids[i] for i, x in enumerate(df_target["searched_ids"].values)]
            
            # def convert_text_id_to_text(text_ids):
            #     return [wiki_dict[text_id]  if text_id != -1 else -1 for text_id in text_ids]
                
            # searched_text_ids, similarity = search(index=index, embeddings=embeddings_target, ids=text_ids, k=30)
            # df_target["searched_text_ids"] = [x + searched_text_ids[i] for i, x in enumerate(df_target["searched_text_ids"].values)]
            # df_target["similarity"] = [x + similarity[i] for i, x in enumerate(df_target["similarity"].values)]
            # df_target["searched_text"] = [x + convert_text_id_to_text(searched_text_ids[i]) for i, x in enumerate(df_target["searched_text"].values)]
            
            # searched_ids, _ = search(index=index, embeddings=embeddings_test, ids=wiki_ids, k=30)
            # df_test["searched_ids"] = [x + searched_ids[i] for i, x in enumerate(df_test["searched_ids"].values)]
            # searched_text_ids, similarity = search(index=index, embeddings=embeddings_test, ids=text_ids, k=30)
            # df_test["searched_text_ids"] = [x + searched_text_ids[i] for i, x in enumerate(df_test["searched_text_ids"].values)]
            # df_test["similarity"] = [x + similarity[i] for i, x in enumerate(df_test["similarity"].values)]
            # df_test["searched_text"] = [x + convert_text_id_to_text(searched_text_ids[i]) for i, x in enumerate(df_test["searched_text"].values)]
        
        logger.info("後処理")

        df_target["similarity"] = [np.array(x) for x in df_target["similarity"].values]
        df_test["similarity"] = [np.array(x) for x in df_test["similarity"].values]
        df_target["similarity_rank"] = [np.argsort(x)[:30] for x in df_target["similarity"].values]
        df_test["similarity_rank"] = [np.argsort(x)[:30] for x in df_test["similarity"].values]
        df_target["searched_text_ids"] = [np.array(x)[idx] for x, idx in zip(df_target["searched_text_ids"].values, df_target["similarity_rank"].values)]
        df_target["searched_ids"] = [np.array(x)[idx] for x, idx in zip(df_target["searched_ids"].values, df_target["similarity_rank"].values)]
        df_test["searched_text_ids"] = [np.array(x)[idx] for x, idx in zip(df_test["searched_text_ids"].values, df_test["similarity_rank"].values)]
        df_test["searched_ids"] = [np.array(x)[idx] for x, idx in zip(df_test["searched_ids"].values, df_test["similarity_rank"].values)]

        df_target["searched_text"] = [np.array(x)[idx] for x, idx in zip(df_target["searched_text"].values, df_target["similarity_rank"].values)]
        df_test["searched_text"] = [np.array(x)[idx] for x, idx in zip(df_test["searched_text"].values, df_test["similarity_rank"].values)]
        
        for i in range(10):
            df_target[f"searched_wiki_id_{i}"] = [x[i] for x in df_target["searched_text"].values]
            df_test[f"searched_wiki_id_{i}"] = [x[i] for x in df_test["searched_text"].values]
        
        df_target[df_target["dataset"] == "train"].to_parquet(f"{output_dir}/train.parquet")
        df_target[df_target["dataset"] == "valid"].to_parquet(f"{output_dir}/valid.parquet")
        df_test.to_parquet(f"{output_dir}/test.parquet")
        
        # map@k, recall@k を計算する
        if "wiki_id" not in df_target.columns:
            logger.info("wiki_id が存在しないため, mapkを計算しません")
        else:
            label_ids = df_target["wiki_id"].values.astype(int).astype(str)
            for k in [1, 3, 5, 10, 30]:
                map_score = mapk(label_ids, df_target["searched_ids"].values, k=k)
                logger.info(f"map{k}: {map_score}")
                wandb.log({f"map{k}": map_score})
                
                recall_score = recallk(label_ids, df_target["searched_ids"].values, k=k)
                logger.info(f"recall{k}: {recall_score}")
                wandb.log({f"recall{k}": recall_score})
            
            # wandb を close する
        wandb.finish()
    except Exception as e:
        # 詳細なエラーログを出力する
        logger.exception(e)
        # wandb.finish()
        raise e
    finally:
        # logger を閉じる
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

if __name__ == "__main__":        
    # wiki_text_preprocess = "all_without_sep"
    # target_text_preprocess = "prompt_and_choice_without_sep"
    # model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    # # model_name = "thenlper/gte-small"  # DEBUG
    # max_length = 192
    # for wiki_file_name in [
    #     "token_length90_stride_sentence3_onlya",
    #     # "token_length200_stride100",
    #     # "token_length400_stride200",
    # ]:
    #     # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    #     exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_{wiki_file_name}_all"
    #     config = Config(
    #         experiment_name=exp_name,
    #         wiki_file_names=glob.glob(f"data/wikipedia/sep_chunk/{wiki_file_name}/chunk1.parquet"),
    #         extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_{wiki_file_name}_all_exp004_sep3",
    #         model_name=model_name,
    #         max_length=max_length,
    #         batch_size=128,
    #         wiki_text_preprocess=wiki_text_preprocess,
    #         target_text_preprocess=target_text_preprocess,
    #         # pretrain_index_dir="output/context_pipeline/stage1/exp006.py/20230915122742_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #         target_texts_name=f"data/gpt_generated_data/only_a_250text_2.csv",
    #         # target_texts_name=f"data/gpt_generated_data/20230917_concatenate.csv",
    #         feature_extractor="sentence_transformer",
    #     )
    #     main(config)

    # wiki_text_preprocess = "all_without_sep"
    # target_text_preprocess = "prompt_and_choice_without_sep"
    # # model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    # model_name = "thenlper/gte-base"  # DEBUG
    # max_length = 192
    # for wiki_file_name in [
    #     "token_length100_stride75",
    #     # "token_length200_stride100",
    #     # "token_length400_stride200",
    # ]:
    #     # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    #     exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_{wiki_file_name}_all"
    #     config = Config(
    #         experiment_name=exp_name,
    #         wiki_file_names=glob.glob(f"data/wikipedia/sep_chunk/{wiki_file_name}/*.parquet"),
    #         extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_{wiki_file_name}_all_exp004_sep3",
    #         model_name=model_name,
    #         max_length=max_length,
    #         batch_size=128,
    #         wiki_text_preprocess=wiki_text_preprocess,
    #         target_text_preprocess=target_text_preprocess,
    #         pretrain_index_dir="output/context_pipeline/stage1/exp007.py/20230917193018_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #         # target_texts_name=f"data/gpt_generated_data/only_a_250text_2.csv",
    #         target_texts_name=f"data/gpt_generated_data/20230918_concatenate.csv",
    #         feature_extractor="sentence_transformer",
    #     )
    #     main(config)

    # wiki_text_preprocess = "all_without_sep"
    # target_text_preprocess = "prompt_and_choice_without_sep"
    # # model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    # model_name = "thenlper/gte-base"  # DEBUG
    # for param in [
    #     ("token_length100_stride75", 160),
    #     ("token_length100_stride50", 160),
    #     ("token_length75_stride40", 128),
    #     ("token_length50_stride25", 96),
    # ]:
    #     wiki_file_name = param[0]
    #     max_length = param[1]
    #     # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    #     exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_{wiki_file_name}_all"
    #     config = Config(
    #         experiment_name=exp_name,
    #         wiki_file_names=glob.glob(f"data/wikipedia/sep3/{wiki_file_name}/a.parquet"),
    #         extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_{wiki_file_name}_only_a_exp004_sep3",
    #         model_name=model_name,
    #         max_length=max_length,
    #         batch_size=128,
    #         wiki_text_preprocess=wiki_text_preprocess,
    #         target_text_preprocess=target_text_preprocess,
    #         target_texts_name=f"data/gpt_generated_data/only_a_250text_2.csv",
    #         feature_extractor="sentence_transformer",
    #     )
    #     main(config)


    # wiki_text_preprocess = "all_without_sep"
    # target_text_preprocess = "prompt_and_choice_without_sep"
    # model_name = "thenlper/gte-base"
    # max_length = 160
    # # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    
    # for wiki_file_name in [
    #     "token_length100_stride75_drop_catTrue",
    #     "token_length100_stride75_drop_catFalse",
    #     "token_length90_stride_sentence3_drop_categoryTrue",
    # ]:
    #     exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_{wiki_file_name}_only_a"
    #     config = Config(
    #         experiment_name=exp_name,
    #         wiki_file_names=glob.glob(f"data/wikipedia/sep4/{wiki_file_name}/a.parquet"),
    #         extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_{wiki_file_name}_only_a_exp008",
    #         model_name=model_name,
    #         max_length=max_length,
    #         batch_size=128,
    #         wiki_text_preprocess=wiki_text_preprocess,
    #         target_text_preprocess=target_text_preprocess,
    #         target_texts_name=f"data/gpt_generated_data/only_a_250text_2.csv",
    #         feature_extractor="sentence_transformer",
    #     )
    #     main(config)


    # wiki_text_preprocess = "all_without_sep"
    # target_text_preprocess = "prompt_and_choice_without_sep"
    # model_name = "thenlper/gte-base"
    # max_length = 192
    # # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    
    # for wiki_file_name in [
    #     "token_length150_stride_sentence5_drop_categoryTrue",
    # ]:
    #     exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_{wiki_file_name}_only_a"
    #     config = Config(
    #         experiment_name=exp_name,
    #         wiki_file_names=glob.glob(f"data/wikipedia/sep4/{wiki_file_name}/a.parquet"),
    #         extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_{wiki_file_name}_only_a_exp008",
    #         model_name=model_name,
    #         max_length=max_length,
    #         batch_size=128,
    #         wiki_text_preprocess=wiki_text_preprocess,
    #         target_text_preprocess=target_text_preprocess,
    #         target_texts_name=f"data/gpt_generated_data/only_a_250text_2.csv",
    #         feature_extractor="sentence_transformer",
    #     )
    #     main(config)

    # wiki_text_preprocess = "all_without_sep"
    # target_text_preprocess = "prompt_and_choice_without_sep"
    # model_name = "thenlper/gte-base"
    # max_length = 256
    # # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    
    # for wiki_file_name in [
    #     "token_length240_stride_sentence8_drop_categoryTrue",
    # ]:
    #     exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_{wiki_file_name}_only_a"
    #     config = Config(
    #         experiment_name=exp_name,
    #         wiki_file_names=glob.glob(f"data/wikipedia/sep4/{wiki_file_name}/a.parquet"),
    #         extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_{wiki_file_name}_only_a_exp008",
    #         model_name=model_name,
    #         max_length=max_length,
    #         batch_size=128,
    #         wiki_text_preprocess=wiki_text_preprocess,
    #         target_text_preprocess=target_text_preprocess,
    #         target_texts_name=f"data/gpt_generated_data/only_a_250text_2.csv",
    #         feature_extractor="sentence_transformer",
    #     )
    #     main(config)

    # wiki_text_preprocess = "all_without_sep"
    # target_text_preprocess = "prompt_and_choice_without_sep"
    # model_name = "thenlper/gte-base"
    # max_length = 160
    # # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    
    # for wiki_file_name in [
    #     "token_length120_stride_sentence4_drop_categoryTrue",
    # ]:
    #     exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_{wiki_file_name}_only_a"
    #     config = Config(
    #         experiment_name=exp_name,
    #         wiki_file_names=glob.glob(f"data/wikipedia/sep4/{wiki_file_name}/a.parquet"),
    #         extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_{wiki_file_name}_only_a_exp008",
    #         model_name=model_name,
    #         max_length=max_length,
    #         batch_size=128,
    #         wiki_text_preprocess=wiki_text_preprocess,
    #         target_text_preprocess=target_text_preprocess,
    #         target_texts_name=f"data/gpt_generated_data/only_a_250text_2.csv",
    #         feature_extractor="sentence_transformer",
    #     )
    #     main(config)
    
    # wiki_text_preprocess = "all_without_sep"
    # target_text_preprocess = "prompt_and_choice_without_sep"
    # model_name = "thenlper/gte-base"
    # max_length = 192
    # # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    
    # for wiki_file_name in [
    #     # "token_length90_stride_sentence3_drop_categoryTrue",
    #     "token_length120_stride_sentence4_drop_categoryTrue",
    #     "token_length150_stride_sentence5_drop_categoryTrue",
    # ]:
    #     exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_{wiki_file_name}_only_a"
    #     config = Config(
    #         experiment_name=exp_name,
    #         wiki_file_names=glob.glob(f"data/wikipedia_fixed/sep_a/{wiki_file_name}/chunk1.parquet"),
    #         extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_{wiki_file_name}_only_a_exp009",
    #         model_name=model_name,
    #         max_length=max_length,
    #         batch_size=128,
    #         wiki_text_preprocess=wiki_text_preprocess,
    #         target_text_preprocess=target_text_preprocess,
    #         target_texts_name=f"data/gpt_generated_data/only_a_250text_3.csv",
    #         feature_extractor="sentence_transformer",
    #     )
    #     main(config)
        
    # wiki_text_preprocess = "all_without_sep"
    # target_text_preprocess = "prompt_and_choice_without_sep"
    # model_name = "thenlper/gte-base"
    # max_length = 192
    # # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    
    # for wiki_file_name in [
    #     "token_length120_stride_sentence4_drop_categoryTrue",
    # ]:
    #     exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_{wiki_file_name}_all"
    #     config = Config(
    #         experiment_name=exp_name,
    #         wiki_file_names=glob.glob(f"data/wikipedia_fixed/sep_chunk/{wiki_file_name}/*.parquet"),
    #         extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_{wiki_file_name}_all_exp009",
    #         model_name=model_name,
    #         max_length=max_length,
    #         batch_size=128,
    #         wiki_text_preprocess=wiki_text_preprocess,
    #         target_text_preprocess=target_text_preprocess,
    #         target_texts_name=f"data/gpt_generated_data/20230922_concatenate.csv",
    #         feature_extractor="sentence_transformer",
    #     )
    #     main(config)
    
    # wiki_text_preprocess = "all_without_sep"
    # target_text_preprocess = "prompt_and_choice_without_sep"
    # model_name = "thenlper/gte-base"
    # max_length = 192
    # # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    # wiki_file_name = "token_length120_stride_sentence4_drop_categoryTrue"
    # for nlists in [1]:
    #     for M in [64]:
    #         exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_{wiki_file_name}_only_a_nlist{nlists}_M{M}"
    #         config = Config(
    #             experiment_name=exp_name,
    #             wiki_file_names=glob.glob(f"data/wikipedia_fixed/sep_a/{wiki_file_name}/chunk1.parquet"),
    #             extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_{wiki_file_name}_only_a_exp009",
    #             model_name=model_name,
    #             max_length=max_length,
    #             batch_size=128,
    #             wiki_text_preprocess=wiki_text_preprocess,
    #             target_text_preprocess=target_text_preprocess,
    #             target_texts_name=f"data/gpt_generated_data/only_a_250text_3.csv",
    #             feature_extractor="sentence_transformer",
    #             faiss_index_parameter={"nlists": nlists, "M": M, "nbits": 8},
    #         )
    #         main(config)
            
            
    # wiki_text_preprocess = "all_without_sep"
    # target_text_preprocess = "prompt_and_choice_without_sep"
    # model_name = "thenlper/gte-base"
    # max_length = 192
    # # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    
    # for wiki_file_name in [
    #     "token_length120_stride_sentence4_drop_categoryTrue",
    # ]:
    #     exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_{wiki_file_name}_all"
    #     config = Config(
    #         experiment_name=exp_name,
    #         wiki_file_names=glob.glob(f"data/wikipedia_fixed/sep_chunk/{wiki_file_name}/*.parquet"),
    #         extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_{wiki_file_name}_all_exp009",
    #         model_name=model_name,
    #         max_length=max_length,
    #         batch_size=128,
    #         wiki_text_preprocess=wiki_text_preprocess,
    #         target_text_preprocess=target_text_preprocess,
    #         target_texts_name=f"data/gpt_generated_data/20230925_concatenate.csv",
    #         # pretrain_index_dir="output/context_pipeline/stage1/exp009.py/20230922162941_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",
    #         feature_extractor="sentence_transformer",
    #     )
    #     main(config)

    wiki_text_preprocess = "all_without_sep"
    target_text_preprocess = "prompt_and_choice_without_sep"
    max_length = 192
    # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    wiki_file_name = "token_length120_stride_sentence4_drop_categoryTrue"
    for model_name in [
        "BAAI/bge-base-en-v1.5",
        "intfloat/e5-base-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ]:
        exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_{wiki_file_name}_only_a"
        config = Config(
            experiment_name=exp_name,
            wiki_file_names=glob.glob(f"data/wikipedia_fixed/sep_a/{wiki_file_name}/chunk1.parquet"),
            extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_{wiki_file_name}_only_a_exp009",
            model_name=model_name,
            max_length=max_length,
            batch_size=128,
            wiki_text_preprocess=wiki_text_preprocess,
            target_text_preprocess=target_text_preprocess,
            target_texts_name=f"data/gpt_generated_data/only_a_250text_3.csv",
            feature_extractor="sentence_transformer",
        )
        main(config)