import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Paths
DATA_DIR = Path("data")
RETRIEVAL_DIR = Path("retrieval")

TRAIN_PATH = DATA_DIR / "train.tsv"
TEST_PATH = DATA_DIR / "test.tsv"
EXTRA_PATH = DATA_DIR / "Datensatz_renamed.csv"

OUTPUT_CSV = DATA_DIR / "Datensatz_final.csv"
FAISS_INDEX_PATH = RETRIEVAL_DIR / "faiss_index_final.bin"


def main():
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_PATH, sep="\t")
    test_df = pd.read_csv(TEST_PATH, sep="\t")
    extra_df = pd.read_csv(EXTRA_PATH)

    common_cols = list(set(train_df.columns) & set(extra_df.columns))
    useful_cols = [c for c in ["claim", "explanation", "label"] if c in common_cols]

    train_df = train_df[useful_cols]
    test_df = test_df[useful_cols]
    extra_df = extra_df[useful_cols]

    full_df = (
        pd.concat([train_df, test_df, extra_df])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    full_df["text"] = (
        full_df["claim"].fillna("") + " " +
        full_df["explanation"].fillna("")
    )

    print(f"Final dataset size: {full_df.shape}")

    print("Generating embeddings...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(
        full_df["text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    RETRIEVAL_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    full_df.to_csv(OUTPUT_CSV, index=False)

    print("FAISS index saved to:", FAISS_INDEX_PATH)
    print("Dataset saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
