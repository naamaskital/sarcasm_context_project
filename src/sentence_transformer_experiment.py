from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42
REPORT_DIR = Path("reports/sample_sentence_transformer")
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


def evaluate_setting(model, setting_name, train_texts, test_texts, y_train, y_test):
    print(f"\nEncoding: {setting_name}")
    train_embeddings = model.encode(
        train_texts.tolist(), batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    test_embeddings = model.encode(
        test_texts.tolist(), batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    classifier.fit(train_embeddings, y_train)
    predictions = classifier.predict(test_embeddings)

    result = {
        "method": "sentence_transformer_embeddings",
        "setting": setting_name,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "sarcastic_f1": f1_score(y_test, predictions, pos_label=1),
    }
    report = classification_report(
        y_test, predictions, target_names=["not_sarcastic", "sarcastic"], digits=4
    )
    return result, predictions, report


def main():
    data_path = find_data_path()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path).dropna(subset=["context", "comment", "label"]).copy()

    df["context_only"] = df["context"].astype(str)
    df["comment_only"] = df["comment"].astype(str)
    df["context_plus_comment"] = (
        "Previous message: " + df["context"].astype(str)
        + "\nReddit comment: " + df["comment"].astype(str)
    )

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    y_train = train_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    print("Data:", data_path)
    print("Loading sentence transformer model...")
    model = SentenceTransformer(MODEL_NAME)

    settings = [
        ("context_only", "context_only"),
        ("comment_only", "comment_only"),
        ("context_plus_comment", "context_plus_comment"),
    ]
    results = []
    comparison_df = test_df[["context", "comment", "label"]].copy()
    reports = {}

    for setting_name, column_name in settings:
        result, predictions, report = evaluate_setting(
            model,
            setting_name,
            train_df[column_name],
            test_df[column_name],
            y_train,
            y_test,
        )
        results.append(result)
        comparison_df[f"prediction_{setting_name}"] = predictions
        reports[setting_name] = report

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(REPORT_DIR / "metrics.csv", index=False)
    comparison_df.to_csv(REPORT_DIR / "predictions.csv", index=False)

    with open(REPORT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Bundled-sample Sentence Transformer sanity check\n")
        f.write("This run is separate from the larger final embedding experiments in scripts/.\n\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")
        for setting_name, report in reports.items():
            f.write(f"{setting_name}\n{'-' * len(setting_name)}\n{report}\n")

    print(metrics_df.to_string(index=False))
    print("Saved to:", REPORT_DIR)


if __name__ == "__main__":
    main()
