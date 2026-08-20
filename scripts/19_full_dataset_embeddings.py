from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.full_dataset_utils import load_or_create_splits

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REPORT_DIR = ROOT / "reports" / "full_dataset" / "embeddings"
CACHE_DIR = ROOT / ".cache" / "sarcasm_context_project" / "full_dataset_embeddings"
BATCH_SIZE = 128
CHUNK_SIZE = 20000
SEED = 42


def summarize(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "sarcastic_f1": float(f1_score(y_true, y_pred, pos_label=1)),
    }


def encode_to_memmap(model, texts, path, dim):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(texts)

    if path.exists():
        print("Using cached embeddings:", path)
        return np.memmap(path, dtype="float32", mode="r", shape=(n, dim))

    mmap = np.memmap(path, dtype="float32", mode="w+", shape=(n, dim))
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        print(f"Encoding {path.stem}: {start:,}-{end:,}/{n:,}")
        emb = model.encode(
            texts.iloc[start:end].tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        mmap[start:end] = emb
        mmap.flush()
    return mmap


def select_features(X_context, X_comment, start, end, mode):
    if mode == "comment_only":
        return np.asarray(X_comment[start:end])
    if mode == "context_only":
        return np.asarray(X_context[start:end])
    if mode == "dual_embeddings":
        return np.concatenate(
            [np.asarray(X_context[start:end]), np.asarray(X_comment[start:end])],
            axis=1,
        )
    raise ValueError(mode)


def train_online_linear_classifier(X_context, X_comment, y, mode):
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        class_weight=None,
        random_state=SEED,
        average=True,
    )
    classes = np.array([0, 1])

    for start in range(0, len(y), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(y))
        X = select_features(X_context, X_comment, start, end, mode)
        clf.partial_fit(X, y[start:end], classes=classes)
        print(f"Training {mode}: {end:,}/{len(y):,}")

    return clf


def predict_in_chunks(clf, X_context, X_comment, mode):
    preds = np.empty(len(X_comment), dtype=np.int64)
    probs = np.empty(len(X_comment), dtype=np.float32)

    for start in range(0, len(preds), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(preds))
        X = select_features(X_context, X_comment, start, end, mode)
        preds[start:end] = clf.predict(X)
        probs[start:end] = clf.predict_proba(X)[:, 1].astype(np.float32)

    return preds, probs


def save_predictions(df, pred, prob, mode, split_name):
    keep = [c for c in ["subreddit", "context", "comment", "label"] if c in df.columns]
    out = df[keep].copy()
    out["prediction"] = pred
    out["prob_sarcastic"] = prob
    out.to_parquet(REPORT_DIR / f"{mode}_{split_name}_predictions.parquet", index=False)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_or_create_splits()
    print("Full-data split sizes:", len(train_df), len(val_df), len(test_df))

    model = SentenceTransformer(MODEL_ID)
    dim = model.get_sentence_embedding_dimension()

    embeddings = {}
    for split_name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        embeddings[(split_name, "context")] = encode_to_memmap(
            model,
            df["context"],
            CACHE_DIR / f"{split_name}_context.f32",
            dim,
        )
        embeddings[(split_name, "comment")] = encode_to_memmap(
            model,
            df["comment"],
            CACHE_DIR / f"{split_name}_comment.f32",
            dim,
        )

    rows = []
    y_train = train_df["label"].to_numpy()

    for mode in ["comment_only", "context_only", "dual_embeddings"]:
        print("\nTraining full-data embedding classifier:", mode)
        clf = train_online_linear_classifier(
            embeddings[("train", "context")],
            embeddings[("train", "comment")],
            y_train,
            mode,
        )

        for split_name, df in [("validation", val_df), ("test", test_df)]:
            pred, prob = predict_in_chunks(
                clf,
                embeddings[(split_name, "context")],
                embeddings[(split_name, "comment")],
                mode,
            )
            y = df["label"].to_numpy()
            row = {
                "model": "all-MiniLM-L6-v2 + online linear classifier",
                "input": mode,
                "split": split_name,
                "n_examples": len(df),
                **summarize(y, pred),
            }
            rows.append(row)
            print(row)
            save_predictions(df, pred, prob, mode, split_name)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORT_DIR / "full_dataset_embedding_metrics.csv", index=False)
    print("\nFINAL")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
