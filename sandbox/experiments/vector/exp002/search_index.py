
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import tqdm
import pickle

def search(query, index, k=3):
    _, searched_index = index.search(query, k)
    return searched_index

# index を load する
index = faiss.read_index("experiments/vector/exp002/output/index.faiss")
# to_gpu
# index = faiss.index_cpu_to_all_gpus(index)

df_train = pd.read_csv("experiments/vector/exp001/input/20230828082546.csv")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# 事前に作成した ids を読み込む
with open("experiments/vector/exp002/output/ids.pickle", 'rb') as f:
    ids = pickle.load(f)
# df_train  prompt に対して, 事前に作成した faiss index を使って類似する prompt を探す
prompt = df_train["prompt"].values
prompt = model.encode(prompt).astype(np.float32)
searched_index = search(prompt, index, k=30)
searched_ids = []
for i in tqdm.tqdm(range(len(searched_index))):
    searched_ids.append([ids[idx] if idx != -1 else -1 for idx in searched_index[i]])

df_train["searched_index"] = searched_ids
df_train.to_csv("output/searched_index.csv", index=False)