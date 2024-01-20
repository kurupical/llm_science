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

from experiments.text_retrieval.exp001 import RetrievalNet, RetrievalDataset, RetrievalConfig
import json

@dataclasses.dataclass
class Config:
    extract_file_name: str  # 作成した embedding を保存するファイル名
    experiment_name: str # 実験名
    wiki_file_name: str = "data/wikipedia/a.parquet"
    target_texts_name: str = "data/gpt_generated_data/only_a_250text.csv"

    # feature extractor
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    feature_extractor: str = "sentence_transformer"
    batch_size: int = 128
    pretrained_model_path: str = None
    
    # text preprocess
    wiki_text_preprocess: str = "all"
    target_text_preprocess: str = "only_prompt"
    debug: bool = False
    max_length: int = 256
    
    # faiss index
    faiss_index: faiss.Index = faiss.IndexIVFPQ
    faiss_index_parameter: dict = dataclasses.field(
        default_factory=lambda: {"nlists": 8, "M": 16, "nbits": 8}
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
        model = pipeline("feature-extraction", model=self.config.model_name)
        embeddings = model(texts, batch_size=self.config.batch_size)
        return embeddings

    def extract_feature(
        self, 
        df: pd.DataFrame,
    ):
        """
        与えられたテキストから embedding を作成する. 
        """
        
        self.logger.info("extract feature start")
        df["texts_concatenate"] = self._preprocess_texts(df)
        texts = df["texts_concatenate"].values.tolist()
        # すでに embedding が作成されている場合は, それを返す
        if os.path.isfile(self.config.extract_file_name) and self.__class__.__name__ == "FeatureExtractorFromWiki":
            self.logger.info(f"すでに embedding が作成されているため, それを返します: {self.config.extract_file_name}")
            return np.load(self.config.extract_file_name)

        if self.config.feature_extractor == "custom_bert":
            embeddings = self._extract_feature_custom_bert(texts)
        elif self.config.feature_extractor == "sentence_transformer":
            embeddings = self._extract_feature_sentence_transformer(texts)
        elif self.config.feature_extractor == "bert":
            embeddings = self._extract_feature_bert(texts)
        else:
            raise ValueError()
        if self.__class__.__name__ == "FeatureExtractorFromWiki":
            self.logger.info(f"embedding を保存します: {self.config.extract_file_name}")
            np.save(self.config.extract_file_name, embeddings)
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
    ):
        def f(text):
            return text.split("==")[0]
        
        if self.text_preprocess == "all":
            return df["title"].astype(str) + " [SEP] " + df["text"]
        if self.text_preprocess == "all_without_sep":
            return df["title"].astype(str) + " " + df["text"]
        if self.text_preprocess == "only_text":
            return df["text"]
        if self.text_preprocess == "title_and_abstract":
            return df["title"].astype(str) + " [SEP] " +  df["text"].astype(str).apply(f)
        raise NotImplementedError(f"想定していないwiki_text_processです: {self.config.text_preprocess}")
        
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
        _, searched_index = index.search(query, k)
        return searched_index
    searched_index = _search(embeddings, index, k=k)
    searched_ids = []
    for i in tqdm.tqdm(range(len(searched_index))):
        searched_ids.append([ids[idx] if idx != -1 else -1 for idx in searched_index[i]])
    return searched_ids


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
        for j in range(k):
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
        
        shutil.copy(__file__, f"{output_dir}/exp002.py")
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
        df_wiki = pd.read_parquet(config.wiki_file_name)
        if config.debug:
            df_wiki = df_wiki.iloc[:1000]
        
        # wiki から embedding を作成する
        feature_extractor = FeatureExtractorFromWiki(config=config, logger=logger)
        embeddings_wiki = feature_extractor.extract_feature(df=df_wiki)
        wiki_dict = dict(zip(df_wiki["id"].values, df_wiki["texts_concatenate"].values))
        
        # embedding から index を作成する    
        index_creator = IndexCreator(config=config, logger=logger)
        index = index_creator.create_index(embeddings=embeddings_wiki)
        
        # faiss index を保存する
        faiss.write_index(index, f"{output_dir}/index.faiss")

        # target_texts から embedding を作成する
        df_target = pd.read_csv(config.target_texts_name)
        feature_extractor = FeatureExtactorFromPrompt(config=config, logger=logger)
        embeddings_target = feature_extractor.extract_feature(df=df_target)
        
        # 事前に作成した faiss index を使って類似する prompt を探す
        searched_ids = search(index=index, embeddings=embeddings_target, ids=df_wiki["id"].values, k=30)
        df_target["searched_ids"] = searched_ids
        
        def convert_wiki_id_to_wiki_text(df):
            for i in range(10):
                df[f"searched_wiki_id_{i}"] = df[f"searched_ids"].apply(lambda x: wiki_dict[x[i]] if x[i] != -1 else -1)            
            return df
            
        df_target = convert_wiki_id_to_wiki_text(df_target)
        df_target.to_parquet(f"{output_dir}/searched_index_train.parquet")
        
        # test data の prompt を探す
        df_test = pd.read_csv("data/kaggle-llm-science-exam/train.csv")
        embeddings_test = feature_extractor.extract_feature(df=df_test)
        
        df_test["searched_ids"] = search(index=index, embeddings=embeddings_test, ids=df_wiki["id"].values, k=30)
        df_test = convert_wiki_id_to_wiki_text(df_test)        
        df_test.to_csv(f"{output_dir}/searched_index_test.csv", index=False)
        
        # map@k, recall@k を計算する
        if "wiki_id" not in df_target.columns:
            logger.info("wiki_id が存在しないため, mapkを計算しません")
        else:
            label_ids = df_target["wiki_id"].values.astype(str)
            for k in [1, 3, 5, 10, 30]:
                map_score = mapk(label_ids, np.array(searched_ids), k=k)
                logger.info(f"map{k}: {map_score}")
                wandb.log({f"map{k}": map_score})
                
                recall_score = recallk(label_ids, np.array(searched_ids), k=k)
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
    # for pretrained_model_path in glob.glob(
    #     "output/text_retrieval/exp001.py/*/model_fold0.pth"
    # ):
    #     wiki_text_preprocess = "all"
    #     # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_only_a_text"
    #     pretrained_name = pretrained_model_path.split("/")[-2]
    #     exp_name = f"{pretrained_name}_wiki{wiki_text_preprocess}_all_text"
    #     config = Config(
    #         experiment_name=exp_name,
    #         pretrained_model_path=pretrained_model_path,
    #         batch_size=512,
    #         feature_extractor="custom_bert",
    #         wiki_file_name="data/wikipedia/a.parquet",
    #         # wiki_file_name="data/wikipedia/all.parquet",
    #         extract_file_name=f"output/embeddings/{pretrained_name}_{wiki_text_preprocess}_only_a.npy",
    #         # extract_file_name=f"output/embeddings/{pretrained_name}_{wiki_text_preprocess}_all.npy",
    #         model_name="sentence-transformers/all-MiniLM-L6-v2",
    #         wiki_text_preprocess=wiki_text_preprocess,
    #         target_text_preprocess="prompt_and_choice",
    #         target_texts_name="data/gpt_generated_data/only_a_250text.csv",
    #     )
    #     main(config)
    # wiki_text_preprocess = "all"
    # target_text_preprocess = "prompt_and_choice"
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # for wiki_file_name in [
    #     "token_length200_stride100",
    #     "token_length100_stride50",
    #     "token_length400_stride200"
    # ]:
    #     # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    #     exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_all"
    #     config = Config(
    #         experiment_name=exp_name,
    #         # wiki_file_name="data/wikipedia/a.parquet",
    #         wiki_file_name=f"data/wikipedia/sep/{wiki_file_name}/all.parquet",
    #         # extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_all.npy",
    #         extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_file_name}_{wiki_text_preprocess}_all.npy",
    #         model_name=model_name,
    #         wiki_text_preprocess=wiki_text_preprocess,
    #         target_text_preprocess=target_text_preprocess,
    #         target_texts_name="data/gpt_generated_data/20230828141216.csv",
    #         feature_extractor="sentence_transformer",
    #     )
    #     main(config)
    
    # wiki_text_preprocess = "only_text"
    # target_text_preprocess = "prompt_and_choice"
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # for target_texts_name in [
    #     # "mmlu_train_fixed",
    #     # "all_12_with_context2_without_generated_gpt",
    #     "20230906_merge_dataset",
    # ]:
    #     # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    #     exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_{target_texts_name}"
    #     config = Config(
    #         experiment_name=exp_name,
    #         # wiki_file_name="data/wikipedia/a.parquet",
    #         wiki_file_name="data/wikipedia/all.parquet",
    #         # extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_all.npy",
    #         extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_all.npy",
    #         model_name=model_name,
    #         wiki_text_preprocess=wiki_text_preprocess,
    #         target_text_preprocess=target_text_preprocess,
    #         target_texts_name=f"data/gpt_generated_data/{target_texts_name}.csv",
    #         feature_extractor="sentence_transformer",
    #     )
    #     main(config)
        
    # wiki_text_preprocess = "only_text"
    # target_text_preprocess = "prompt_and_choice"
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # for max_length in [192, 256, 384, 512]:
    #     # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    #     exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_maxlength{max_length}_only_a"
    #     config = Config(
    #         experiment_name=exp_name,
    #         wiki_file_name="data/wikipedia/a.parquet",
    #         # wiki_file_name="data/wikipedia/all.parquet",
    #         extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_only_a.npy",
    #         # extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_all.npy",
    #         model_name=model_name,
    #         max_length=max_length,
    #         wiki_text_preprocess=wiki_text_preprocess,
    #         target_text_preprocess=target_text_preprocess,
    #         target_texts_name=f"data/gpt_generated_data/only_a_250text_2.csv",
    #         feature_extractor="sentence_transformer",
    #     )
    #     main(config)
        
    # wiki_text_preprocess = "only_text"
    # target_text_preprocess = "prompt_and_choice_without_sep"
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # for max_length in [192, 256, 384, 512]:
    #     # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
    #     exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_maxlength{max_length}_only_a"
    #     config = Config(
    #         experiment_name=exp_name,
    #         wiki_file_name="data/wikipedia/a.parquet",
    #         # wiki_file_name="data/wikipedia/all.parquet",
    #         extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_only_a.npy",
    #         # extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_all.npy",
    #         model_name=model_name,
    #         max_length=max_length,
    #         wiki_text_preprocess=wiki_text_preprocess,
    #         target_text_preprocess=target_text_preprocess,
    #         target_texts_name=f"data/gpt_generated_data/only_a_250text_2.csv",
    #         feature_extractor="sentence_transformer",
    #     )
    #     main(config)
        
    wiki_text_preprocess = "all_without_sep"
    target_text_preprocess = "prompt_and_choice_without_sep"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    for max_length in [192, 256, 384, 512]:
        # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
        exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_maxlength{max_length}_only_a"
        config = Config(
            experiment_name=exp_name,
            wiki_file_name="data/wikipedia/a.parquet",
            # wiki_file_name="data/wikipedia/all.parquet",
            extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_only_a.npy",
            # extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_all.npy",
            model_name=model_name,
            max_length=max_length,
            wiki_text_preprocess=wiki_text_preprocess,
            target_text_preprocess=target_text_preprocess,
            target_texts_name=f"data/gpt_generated_data/only_a_250text_2.csv",
            feature_extractor="sentence_transformer",
        )
        main(config)
    wiki_text_preprocess = "all_without_sep"
    target_text_preprocess = "prompt_and_choice_without_sep"
    model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    
    for max_length in [192]:
        # exp_name = f"{model_name}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_only_a_text"
        exp_name = f"{os.path.basename(model_name)}_wiki{wiki_text_preprocess}_target{target_text_preprocess}_maxlength{max_length}_only_a"
        config = Config(
            experiment_name=exp_name,
            wiki_file_name="data/wikipedia/a.parquet",
            # wiki_file_name="data/wikipedia/all.parquet",
            extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_{max_length}_only_a.npy",
            # extract_file_name=f"output/embeddings/{os.path.basename(model_name)}_{wiki_text_preprocess}_all.npy",
            model_name=model_name,
            max_length=max_length,
            wiki_text_preprocess=wiki_text_preprocess,
            target_text_preprocess=target_text_preprocess,
            target_texts_name=f"data/gpt_generated_data/only_a_250text_2.csv",
            feature_extractor="sentence_transformer",
        )
        main(config)