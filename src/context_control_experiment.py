from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
REPORT_DIR = Path("reports/sample_baseline")
DATA_CANDIDATES = [
    Path("data/reddit_sarcasm_context_sample.csv"),
    Path("data_backup/reddit_sarcasm_context_sample.csv"),
]


def find_data_path():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find the bundled sample. Expected data/reddit_sarcasm_context_sample.csv "
        "or data_backup/reddit_sarcasm_context_sample.csv"
    )


def make_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=30000,
            min_df=2,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def evaluate(setting, train_texts, test_texts, y_train, y_test):
    model = make_model()
    model.fit(train_texts, y_train)
    predictions = model.predict(test_texts)
    return {
        "setting": setting,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "sarcastic_f1": f1_score(y_test, predictions, pos_label=1),
    }, predictions, classification_report(
        y_test,
        predictions,
        target_names=["not_sarcastic", "sarcastic"],
        digits=4,
    )


def combine(contexts, comments):
    contexts = pd.Series(np.asarray(contexts), index=comments.index).astype(str)
    return "Previous message: " + contexts + "\nReddit comment: " + comments.astype(str)


def main():
    data_path = find_data_path()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path).dropna(subset=["context", "comment", "label"]).copy()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    train_df = train_df.copy()
    test_df = test_df.copy()

    rng = np.random.default_rng(RANDOM_STATE)
    train_random = train_df["context"].to_numpy().copy()
    test_random = test_df["context"].to_numpy().copy()
    rng.shuffle(train_random)
    rng.shuffle(test_random)
    train_random = np.roll(train_random, 1)
    test_random = np.roll(test_random, 1)

    settings = {
        "context_only": (
            train_df["context"].astype(str),
            test_df["context"].astype(str),
        ),
        "comment_only": (
            train_df["comment"].astype(str),
            test_df["comment"].astype(str),
        ),
        "true_context_plus_comment": (
            combine(train_df["context"], train_df["comment"]),
            combine(test_df["context"], test_df["comment"]),
        ),
        "random_context_plus_comment": (
            combine(train_random, train_df["comment"]),
            combine(test_random, test_df["comment"]),
        ),
    }

    results = []
    prediction_table = test_df[["context", "comment", "label"]].copy()
    reports = {}

    for name, (train_texts, test_texts) in settings.items():
        result, predictions, report = evaluate(
            name,
            train_texts,
            test_texts,
            train_df["label"],
            test_df["label"],
        )
        results.append(result)
        prediction_table[f"prediction_{name}"] = predictions
        reports[name] = report

    metrics = pd.DataFrame(results)
    metrics.to_csv(REPORT_DIR / "metrics.csv", index=False)
    prediction_table.to_csv(REPORT_DIR / "predictions.csv", index=False)

    with open(REPORT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Bundled 2,000-example sample sanity check\n")
        f.write("This run is separate from the larger final experiments reported in reports/final_results.csv.\n\n")
        f.write(metrics.to_string(index=False))
        f.write("\n\n")
        for name, report in reports.items():
            f.write(f"{name}\n{'-' * len(name)}\n{report}\n")

    print(f"Data: {data_path}")
    print(metrics.to_string(index=False))
    print(f"Saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
