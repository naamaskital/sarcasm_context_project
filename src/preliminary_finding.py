import os
import re
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


DATASET_NAME = "marcbishara/sarcasm-on-reddit"
DATASET_SPLIT = "sft_train"
TARGET_PER_CLASS = 1000
RANDOM_STATE = 42

os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def clean_text(value):
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"[deleted]", "[removed]", "nan", "none"}:
        return ""

    return text


def normalize_label(value):
    if value is None:
        return None

    text = str(value).strip().lower()

    if text in {"1", "sarcastic", "true", "yes"}:
        return 1

    if text in {"0", "not_sarcastic", "not sarcastic", "false", "no"}:
        return 0

    try:
        number = int(value)
        if number in {0, 1}:
            return number
    except Exception:
        pass

    return None


def get_first_existing(example, names):
    for name in names:
        if name in example and example[name] is not None:
            return example[name]
    return None


def parse_text_field(text):
    """
    Tries to extract:
    - context from "User: ..."
    - reply/comment from "Reddit Comment: ..."
    """

    text = clean_text(text)

    user_match = re.search(
        r"User:\s*(.*?)(?:\n\s*Reddit Comment:|Reddit Comment:)",
        text,
        flags=re.DOTALL
    )

    comment_match = re.search(
        r"Reddit Comment:\s*(.*?)(?:<\|endoftext\|>|$)",
        text,
        flags=re.DOTALL
    )

    context = clean_text(user_match.group(1)) if user_match else ""
    comment = clean_text(comment_match.group(1)) if comment_match else ""

    return context, comment


def collect_balanced_sample():
    print("Loading dataset in streaming mode...")

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)

    rows = []
    counts = {0: 0, 1: 0}

    for example in dataset:
        label_value = get_first_existing(
            example,
            ["label", "labels", "is_sarcastic", "sarcasm"]
        )

        label = normalize_label(label_value)

        context = clean_text(get_first_existing(
            example,
            ["parent_comment", "context", "parent", "user"]
        ))

        comment = clean_text(get_first_existing(
            example,
            ["comment", "reply", "response", "reddit_comment"]
        ))

        if not context or not comment:
            text_value = get_first_existing(example, ["text", "input", "prompt"])
            parsed_context, parsed_comment = parse_text_field(text_value)

            if not context:
                context = parsed_context

            if not comment:
                comment = parsed_comment

        if label is None:
            continue

        if not context or not comment:
            continue

        if counts[label] >= TARGET_PER_CLASS:
            continue

        rows.append({
            "context": context,
            "comment": comment,
            "label": label
        })

        counts[label] += 1

        if counts[0] >= TARGET_PER_CLASS and counts[1] >= TARGET_PER_CLASS:
            break

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No usable rows were collected. We need to inspect the dataset columns."
        )

    print("\nCollected rows:")
    print(df["label"].value_counts().rename(index={
        0: "not_sarcastic",
        1: "sarcastic"
    }))

    df.to_csv("data/reddit_sarcasm_context_sample.csv", index=False)

    print("\nExample row:")
    print(df.iloc[0])

    return df


def evaluate_model(name, train_texts, test_texts, y_train, y_test):
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

    result = {
        "setting": name,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "sarcastic_f1": f1_score(y_test, predictions, pos_label=1)
    }

    report = classification_report(
        y_test,
        predictions,
        target_names=["not_sarcastic", "sarcastic"]
    )

    return predictions, result, report


def main():
    df = collect_balanced_sample()

    df["text_without_context"] = df["comment"]

    df["text_with_context"] = (
        "Previous message: " + df["context"] +
        "\nReddit comment: " + df["comment"]
    )

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    y_train = train_df["label"]
    y_test = test_df["label"]

    pred_without_context, result_without_context, report_without_context = evaluate_model(
        "comment_only",
        train_df["text_without_context"],
        test_df["text_without_context"],
        y_train,
        y_test
    )

    pred_with_context, result_with_context, report_with_context = evaluate_model(
        "context_plus_comment",
        train_df["text_with_context"],
        test_df["text_with_context"],
        y_train,
        y_test
    )

    metrics_df = pd.DataFrame([
        result_without_context,
        result_with_context
    ])

    metrics_df.to_csv("reports/preliminary_metrics.csv", index=False)

    comparison_df = test_df[["context", "comment", "label"]].copy()
    comparison_df["prediction_without_context"] = pred_without_context
    comparison_df["prediction_with_context"] = pred_with_context
    comparison_df["changed_prediction"] = (
        comparison_df["prediction_without_context"]
        != comparison_df["prediction_with_context"]
    )

    comparison_df.to_csv("reports/prediction_comparison.csv", index=False)

    changed_df = comparison_df[comparison_df["changed_prediction"]]
    changed_df.to_csv("reports/changed_predictions.csv", index=False)

    with open("reports/summary.txt", "w", encoding="utf-8") as file:
        file.write("Preliminary finding: context and sarcasm detection\n")
        file.write("=" * 55 + "\n\n")

        file.write("Metrics:\n")
        file.write(metrics_df.to_string(index=False))
        file.write("\n\n")

        file.write("Classification report - comment only:\n")
        file.write(report_without_context)
        file.write("\n\n")

        file.write("Classification report - context + comment:\n")
        file.write(report_with_context)
        file.write("\n\n")

        file.write(f"Number of test examples: {len(test_df)}\n")
        file.write(f"Number of predictions changed by adding context: {len(changed_df)}\n")

    print("\nDone.")
    print("\nMetrics:")
    print(metrics_df)

    print("\nFiles created:")
    print("data/reddit_sarcasm_context_sample.csv")
    print("reports/preliminary_metrics.csv")
    print("reports/prediction_comparison.csv")
    print("reports/changed_predictions.csv")
    print("reports/summary.txt")


if __name__ == "__main__":
    main()
