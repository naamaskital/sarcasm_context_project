from pathlib import Path
import json
import sys

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.full_dataset_utils import load_full_corpus

SEED = 42
CACHE_DIR = ROOT / ".cache" / "sarcasm_context_project" / "subreddit_generalization"


def make_group_splits(df):
    if "subreddit" not in df.columns:
        raise ValueError("Dataset does not contain a subreddit column.")

    df = df[df["subreddit"].notna()].copy()
    df["subreddit"] = df["subreddit"].astype(str)

    outer = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    train_val_idx, test_idx = next(outer.split(df, groups=df["subreddit"]))
    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    inner = GroupShuffleSplit(n_splits=1, test_size=1 / 9, random_state=SEED + 1)
    train_idx, val_idx = next(inner.split(train_val, groups=train_val["subreddit"]))
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)

    train_groups = set(train["subreddit"])
    val_groups = set(val["subreddit"])
    test_groups = set(test["subreddit"])

    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise AssertionError("Subreddit leakage detected between group splits.")

    return train, val, test


def load_or_create_subreddit_splits():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": CACHE_DIR / "train.parquet",
        "validation": CACHE_DIR / "validation.parquet",
        "test": CACHE_DIR / "test.parquet",
    }

    if all(p.exists() for p in paths.values()):
        return tuple(pd.read_parquet(paths[name]) for name in ["train", "validation", "test"])

    df = load_full_corpus()
    train, val, test = make_group_splits(df)

    for name, split in [("train", train), ("validation", val), ("test", test)]:
        split.to_parquet(paths[name], index=False)

    metadata = {
        "seed": SEED,
        "train_examples": len(train),
        "validation_examples": len(val),
        "test_examples": len(test),
        "train_subreddits": train["subreddit"].nunique(),
        "validation_subreddits": val["subreddit"].nunique(),
        "test_subreddits": test["subreddit"].nunique(),
        "train_label_counts": train["label"].value_counts().sort_index().to_dict(),
        "validation_label_counts": val["label"].value_counts().sort_index().to_dict(),
        "test_label_counts": test["label"].value_counts().sort_index().to_dict(),
    }
    (CACHE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    return train, val, test


if __name__ == "__main__":
    load_or_create_subreddit_splits()
