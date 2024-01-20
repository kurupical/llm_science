import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import tqdm
import pickle

def search(query, index, k=3):
    _, searched_index = index.search(query, k)
    return searched_index



# 事前に作成した embeddings を読み込み, faiss index(IndexIVFFlat) を作成する
res = faiss.StandardGpuResources()
embeddings = np.load("experiments/vector/exp001/output/concat.npy")
embeddings = embeddings.astype(np.float32)
quantizer = faiss.IndexFlatIP(embeddings.shape[1])
index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], 100)
# index = faiss.index_cpu_to_all_gpus(res, 0, index)

print("train")
index.train(embeddings)
print("add")
index.add(embeddings)

# index を 保存する
faiss.write_index(index, "experiments/vector/exp001/output/index.faiss")
