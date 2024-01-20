from sentence_transformers import SentenceTransformer
import pandas as pd
import tqdm
import glob
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def main():
    # files = glob.glob("../../../data/wikipedia/*.parquet")
    files = glob.glob("data/wikipedia/*.parquet")[:2]
    output_dir = f"experiments/vector/{os.path.basename(__file__).replace('.py', '')}/output"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.concat([pd.read_parquet(f) for f in tqdm.tqdm(files)])
    df["texts_concatenate"] = df["title"].astype(str) + df["text"].astype(str)   
    # TF-IDF Vectorizer で embedding を作成する
    vectorizer = TfidfVectorizer(ngram_range=(2))
    X = vectorizer.fit_transform(df["texts_concatenate"])
    
    # vectorizer を保存する
    with open(f"{output_dir}/vectorizer.pickle", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # TF-IDF Vectorizer で作成した embedding を保存する
    np.save(f"{output_dir}/tfidf.npy", X.toarray().astype(np.float32))


if __name__ == "__main__":
    main()