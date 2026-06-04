import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


INPUT_PATH = "data/reddit_sarcasm_context_sample.csv"
RANDOM_STATE = 42

os.makedirs("reports", exist_ok=True)


def evaluate_model(setting_name, train_texts, test_texts, y_train, y_test):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=30000,
            min_df=2
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(train_texts, y_train)
    predictions = model.predict(test_texts)

    return {
        "setting": setting_name,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "sarcastic_f1": f1_score(y_test, predictions, pos_label=1),
    }, predictions, classification_report(
        y_test,
        predictions,
        target_names=["not_sarcastic", "sarcastic"]
    )


def label_name(value):
    return "sarcastic" if int(value) == 1 else "not_sarcastic"


def main():
    df = pd.read_csv(INPUT_PATH)

    df["context_only"] = df["context"].astype(str)
    df["comment_only"] = df["comment"].astype(str)
    df["context_plus_comment"] = (
        "Previous message: " + df["context"].astype(str) +
        "\nReddit comment: " + df["comment"].astype(str)
    )

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    y_train = train_df["label"]
    y_test = test_df["label"]

    settings = [
        ("context_only", "context_only"),
        ("comment_only", "comment_only"),
        ("context_plus_comment", "context_plus_comment"),
    ]

    results = []
    predictions_by_setting = {}
    reports = {}

    for setting_name, column_name in settings:
        result, predictions, report = evaluate_model(
            setting_name,
            train_df[column_name],
            test_df[column_name],
            y_train,
            y_test
        )

        results.append(result)
        predictions_by_setting[setting_name] = predictions
        reports[setting_name] = report

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("reports/context_control_metrics.csv", index=False)

    comparison_df = test_df[["context", "comment", "label"]].copy()

    for setting_name in predictions_by_setting:
        comparison_df[f"prediction_{setting_name}"] = predictions_by_setting[setting_name]

    comparison_df.to_csv("reports/context_control_predictions.csv", index=False)

    with open("reports/context_control_summary.txt", "w", encoding="utf-8") as f:
        f.write("Context control experiment\n")
        f.write("=" * 40 + "\n\n")

        f.write("Goal:\n")
        f.write(
            "Compare three inputs: context only, comment only, and context + comment.\n"
            "This checks whether the context alone is useful, or whether it helps mainly together with the comment.\n\n"
        )

        f.write("Metrics:\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")

        for setting_name in reports:
            f.write(f"Classification report - {setting_name}:\n")
            f.write(reports[setting_name])
            f.write("\n\n")

        best = metrics_df.sort_values("macro_f1", ascending=False).iloc[0]
        f.write("Best setting by Macro-F1:\n")
        f.write(f"{best['setting']} with Macro-F1 = {best['macro_f1']:.4f}\n\n")

        f.write("Short interpretation template:\n")
        f.write(
            "If context_plus_comment is best, this suggests that context helps when combined with the target comment.\n"
            "If context_only is also strong, the previous message may contain useful hints or dataset-specific signals.\n"
            "If comment_only is close to context_plus_comment, the gain from context exists but is limited for this simple baseline.\n"
        )

    print("Done.")
    print(metrics_df)
    print("\nFiles created:")
    print("reports/context_control_metrics.csv")
    print("reports/context_control_predictions.csv")
    print("reports/context_control_summary.txt")


if __name__ == "__main__":
    main()
