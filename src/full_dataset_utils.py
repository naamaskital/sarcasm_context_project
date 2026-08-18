from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

DATASET_REPO = "marcbishara/sarcasm-on-reddit"
DATASET_FILE = "train-balanced-sarcasm.csv"
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / ".cache" / "sarcasm_context_project"
SPLIT_DIR = CACHE_ROOT / "full_dataset_splits"


def _clean(df):
    context_col = "parent_comment" if "parent_comment" in df.columns else "context"
    needed = ["label", "comment", context_col]
    optional = [c for c in ["author", "subreddit", "created_utc", "date"] if c in df.columns]

    df = df[needed + optional].copy()
    df = df.rename(columns={context_col: "context"})
    df["comment"] = df["comment"].fillna("").astype(str).str.strip()
    df["context"] = df["context"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[(df["comment"] != "") & (df["context"] != "")].reset_index(drop=True)
    return df


def load_full_corpus():
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
    )
    print("Reading full corpus from:", path)
    df = pd.read_csv(path)
    print("Raw rows:", len(df))
    df = _clean(df)
    print("Usable rows after removing missing/empty text:", len(df))
    print("Label counts:", df["label"].value_counts().to_dict())
    return df


def make_fixed_splits(df, train_size=0.80, val_size=0.10, test_size=0.10):
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1")

    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=SEED,
        stratify=df["label"],
    )

    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_or_create_splits():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": SPLIT_DIR / "train.parquet",
        "validation": SPLIT_DIR / "validation.parquet",
        "test": SPLIT_DIR / "test.parquet",
    }

    if all(path.exists() for path in paths.values()):
        print("Loading cached fixed splits from:", SPLIT_DIR)
        return tuple(pd.read_parquet(paths[name]) for name in ["train", "validation", "test"])

    df = load_full_corpus()
    train_df, val_df, test_df = make_fixed_splits(df)

    train_df.to_parquet(paths["train"], index=False)
    val_df.to_parquet(paths["validation"], index=False)
    test_df.to_parquet(paths["test"], index=False)

    print("Fixed split sizes:")
    print("train:", len(train_df))
    print("validation:", len(val_df))
    print("test:", len(test_df))
    print("total:", len(train_df) + len(val_df) + len(test_df))
    print("Saved fixed splits to:", SPLIT_DIR)

    return train_df, val_df, test_df


if __name__ == "__main__":
    load_or_create_splits()
