from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.subreddit_generalization_utils import load_or_create_subreddit_splits

REPORT_DIR = ROOT / "reports" / "subreddit_generalization"
CACHE_DIR = ROOT / ".cache" / "sarcasm_context_project" / "subreddit_generalization_embeddings"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 20000
BATCH_SIZE = 128
SEED = 42


def metrics(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "sarcastic_f1": float(f1_score(y, pred, pos_label=1)),
    }


def text_for_mode(df, mode):
    if mode == "comment_only":
        return df["comment"].astype(str)
    if mode == "context_plus_comment":
        return "Context: " + df["context"].astype(str) + " [SEP] Reply: " + df["comment"].astype(str)
    raise ValueError(mode)


def tfidf_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=120000,
            sublinear_tf=True, dtype=np.float32,
        )),
        ("classifier", LogisticRegression(
            max_iter=100, tol=1e-3, class_weight="balanced",
            solver="saga", random_state=SEED,
        )),
    ])


def encode_to_memmap(model, texts, path, dim):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    if path.exists():
        return np.memmap(path, dtype="float32", mode="r", shape=(n, dim))
    mmap = np.memmap(path, dtype="float32", mode="w+", shape=(n, dim))
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        print(f"Encoding {path.stem}: {start:,}-{end:,}/{n:,}")
        mmap[start:end] = model.encode(
            texts.iloc[start:end].tolist(), batch_size=BATCH_SIZE,
            normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        mmap.flush()
    return mmap


def features(context_emb, comment_emb, start, end, mode):
    if mode == "comment_only":
        return np.asarray(comment_emb[start:end])
    if mode == "dual_embeddings":
        return np.concatenate([
            np.asarray(context_emb[start:end]),
            np.asarray(comment_emb[start:end]),
        ], axis=1)
    raise ValueError(mode)


def train_minilm(context_emb, comment_emb, y, mode):
    clf = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=1e-5,
        random_state=SEED, average=True,
    )
    for start in range(0, len(y), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(y))
        clf.partial_fit(
            features(context_emb, comment_emb, start, end, mode),
            y[start:end], classes=np.array([0, 1]),
        )
    return clf


def predict_minilm(clf, context_emb, comment_emb, mode):
    pred = np.empty(len(comment_emb), dtype=np.int64)
    for start in range(0, len(pred), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(pred))
        pred[start:end] = clf.predict(features(context_emb, comment_emb, start, end, mode))
    return pred


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = load_or_create_subreddit_splits()

    print("Unseen-subreddit split:")
    print("train:", len(train), "subreddits:", train["subreddit"].nunique())
    print("validation:", len(val), "subreddits:", val["subreddit"].nunique())
    print("test:", len(test), "subreddits:", test["subreddit"].nunique())

    rows = []
    y_train = train["label"].to_numpy()

    for mode in ["comment_only", "context_plus_comment"]:
        print("\nTF-IDF:", mode)
        model = tfidf_model()
        model.fit(text_for_mode(train, mode), y_train)
        for split_name, df in [("validation", val), ("test", test)]:
            pred = model.predict(text_for_mode(df, mode))
            rows.append({
                "protocol": "unseen_subreddit",
                "model": "TF-IDF + Logistic Regression",
                "input": mode,
                "split": split_name,
                "n_examples": len(df),
                "n_subreddits": df["subreddit"].nunique(),
                **metrics(df["label"].to_numpy(), pred),
            })

    encoder = SentenceTransformer(MODEL_ID)
    dim = encoder.get_sentence_embedding_dimension()
    emb = {}
    for split_name, df in [("train", train), ("validation", val), ("test", test)]:
        emb[(split_name, "context")] = encode_to_memmap(
            encoder, df["context"], CACHE_DIR / f"{split_name}_context.f32", dim
        )
        emb[(split_name, "comment")] = encode_to_memmap(
            encoder, df["comment"], CACHE_DIR / f"{split_name}_comment.f32", dim
        )

    for mode in ["comment_only", "dual_embeddings"]:
        print("\nMiniLM:", mode)
        clf = train_minilm(emb[("train", "context")], emb[("train", "comment")], y_train, mode)
        for split_name, df in [("validation", val), ("test", test)]:
            pred = predict_minilm(clf, emb[(split_name, "context")], emb[(split_name, "comment")], mode)
            rows.append({
                "protocol": "unseen_subreddit",
                "model": "all-MiniLM-L6-v2 + online linear classifier",
                "input": mode,
                "split": split_name,
                "n_examples": len(df),
                "n_subreddits": df["subreddit"].nunique(),
                **metrics(df["label"].to_numpy(), pred),
            })

    results = pd.DataFrame(rows)
    results.to_csv(REPORT_DIR / "unseen_subreddit_metrics.csv", index=False)

    test_rows = results[results["split"] == "test"].copy()
    gains = []
    for model_name, group in test_rows.groupby("model"):
        by_input = group.set_index("input")
        comment_key = "comment_only"
        context_key = "context_plus_comment" if "context_plus_comment" in by_input.index else "dual_embeddings"
        if comment_key in by_input.index and context_key in by_input.index:
            gains.append({
                "model": model_name,
                "comment_macro_f1": float(by_input.loc[comment_key, "macro_f1"]),
                "context_macro_f1": float(by_input.loc[context_key, "macro_f1"]),
                "context_gain_unseen_subreddits": float(
                    by_input.loc[context_key, "macro_f1"] - by_input.loc[comment_key, "macro_f1"]
                ),
            })
    gains_df = pd.DataFrame(gains)
    gains_df.to_csv(REPORT_DIR / "unseen_subreddit_context_gain.csv", index=False)

    print("\nFINAL UNSEEN-SUBREDDIT RESULTS")
    print(results.to_string(index=False))
    print("\nCONTEXT GAIN ON UNSEEN SUBREDDITS")
    print(gains_df.to_string(index=False))


if __name__ == "__main__":
    main()
