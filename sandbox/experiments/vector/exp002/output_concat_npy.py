import numpy as np
import glob
import os
import pandas as pd
import tqdm
import pickle

files = glob.glob("data/wikipedia/*.parquet")

# faiss index を作成する
ids = []
ary = []
for f in tqdm.tqdm(files):
    fname = os.path.basename(f).replace('.parquet', '')
    npy_fname = f"experiments/vector/exp002/output/{fname}.npy"
    if not os.path.isfile(npy_fname):
        print("npy file not found:", npy_fname)
        continue
    
    ids.extend(pd.read_parquet(f)["id"].values.tolist())
    ary.append(np.load(npy_fname)) 

ary = np.concatenate(ary)
np.save("experiments/vector/exp002/output/concat.npy", ary)

with open("experiments/vector/exp002/output/ids.pickle", 'wb') as f:
    pickle.dump(ids, f)
