# exp006 
import dataclasses
from typing import List, Tuple
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

import ctypes
libc = ctypes.CDLL("libc.so.6")

@dataclasses.dataclass
class Config:
    experiment_name: str # 実験名
    extract_file_names: Tuple[str]  # 作成した embedding を保存するファイル名
    model_names: Tuple[str]
    wiki_file_names: Tuple[str]
    target_texts_name: str = "data/gpt_generated_data/only_a_250text.csv"

    # feature extractor
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
        ret = []
        for model_name in self.config.model_names:
            model = SentenceTransformer(model_name)
            if self.config.max_length is not None:
                model.max_seq_length = self.config.max_length
            
            # 1万件ずつに分割して計算する
            embeddings = []
            for i in tqdm.tqdm(range(0, len(texts), 10000), desc="extract feature"):
                embeddings.append(
                    model.encode(texts[i:i+10000], batch_size=self.config.batch_size, show_progress_bar=False)
                )
            embeddings = np.concatenate(embeddings)
            ret.append(embeddings)
        ret = np.concatenate(ret, axis=1)
        self.logger.info(f"embedding の shape: {ret.shape}")
        return ret
    
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


logger = get_logger(output_dir="")
logger.info("load data")

wiki_text_preprocess = "all_without_sep"
target_text_preprocess = "prompt_and_choice_without_sep"
model_name = sys.argv[3]
max_length = 192 
exp_name = ""
config = Config(
    experiment_name=exp_name,
    wiki_file_names=[sys.argv[1]],
    extract_file_name=f"",
    model_name=model_name,
    max_length=max_length,
    batch_size=128,
    wiki_text_preprocess=wiki_text_preprocess,
    target_text_preprocess=target_text_preprocess,
    pretrain_index_dir=sys.argv[2],
    target_texts_name=f"",
    feature_extractor="sentence_transformer",
)

feature_extractor = FeatureExtactorFromPrompt(config=config, logger=logger)

df_test = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/test.csv")
embeddings_test = feature_extractor.extract_feature(df=df_test)

df_test["searched_text_ids"] = [[]] * len(df_test) 
df_test["searched_text"] = [[]] * len(df_test) 
df_test["similarity"] = [[]] * len(df_test)

del feature_extractor; gc.collect()
libc.malloc_trim(0)
torch.cuda.empty_cache()

text_id_start = 0
for file in config.wiki_file_names:
    logger.info(f"load file {file}")
    df_wiki = pd.read_parquet(file)
    feature_extractor = FeatureExtractorFromWiki(config=config, logger=logger)    
    text_ids = np.arange(len(df_wiki)) + text_id_start
    text_id_start += len(df_wiki)
    wiki_dict = dict(zip(text_ids, feature_extractor._preprocess_texts(df_wiki).values.tolist()))    

    # release memory
    del df_wiki, feature_extractor; gc.collect()
    libc.malloc_trim(0)
    torch.cuda.empty_cache()

    logger.info(f"create index")
    index = faiss.read_index(f"{config.pretrain_index_dir}/index_{os.path.basename(file).split('.')[0]}.faiss")
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    index = faiss.index_cpu_to_gpu(res, 0, index, co)

    def convert_text_id_to_text(text_ids):
        return [wiki_dict[text_id]  if text_id != -1 else -1 for text_id in text_ids]

    logger.info(f"search")
    searched_text_ids, similarity = search(index=index, embeddings=embeddings_test, ids=text_ids, k=30)
    logger.info(f"save result")
    df_test["searched_text_ids"] = [x + searched_text_ids[i] for i, x in enumerate(df_test["searched_text_ids"].values)]
    df_test["similarity"] = [x + similarity[i] for i, x in enumerate(df_test["similarity"].values)]
    df_test["searched_text"] = [x + convert_text_id_to_text(searched_text_ids[i]) for i, x in enumerate(df_test["searched_text"].values)]
    
    del wiki_dict; gc.collect()
    del index, searched_text_ids, similarity; gc.collect()
    logger.info(f"finish {file}")
df_test.to_parquet(os.path.basename(sys.argv[1]))