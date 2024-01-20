from sentence_transformers import SentenceTransformer
import pandas as pd
import tqdm
import glob
import numpy as np
import os

# CLS は最初に出ることを確認
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
# print(tokenizer("Hello world")["input_ids"])


def extract_embeddings(df, desc, batch_size=32):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    df["texts_concatenate"] = df["title"].astype(str) + "[SEP]" + df["text"].astype(str)
    texts = df["texts_concatenate"].values.tolist()

    num_iterations = len(texts) // batch_size
    if len(texts) % batch_size != 0:
        num_iterations += 1

    embeddings = []
    for i in tqdm.tqdm(range(num_iterations), desc=desc):
        texts_ = texts[i*batch_size:(i+1)*batch_size]
        bs = len(texts_)
        feature = model.encode(texts_, batch_size=bs).astype(np.float32)
        embeddings.append(feature)

    return np.concatenate(embeddings)


def main():
    files = glob.glob("../../../data/wikipedia/*.parquet")
    # files = glob.glob("data/wikipedia/*.parquet")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for f in files:
        if os.path.isfile(f"{output_dir}/{os.path.basename(f).replace('.parquet', '')}.npy"):
            continue
        df = pd.read_parquet(f)
        embeddings = extract_embeddings(df, desc=f)
        np.save(f"{output_dir}/{os.path.basename(f).replace('.parquet', '')}.npy", embeddings)


if __name__ == "__main__":
    main()