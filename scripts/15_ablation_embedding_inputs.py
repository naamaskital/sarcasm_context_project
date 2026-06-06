import importlib.util
import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


spec = importlib.util.spec_from_file_location(
    "contrast_script",
    "scripts/13_train_contrast_features.py"
)
contrast_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contrast_script)

RANDOM_STATE = contrast_script.RANDOM_STATE
EMBEDDING_MODEL = contrast_script.EMBEDDING_MODEL

os.makedirs("reports", exist_ok=True)


def encode_texts(model, texts):
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def evaluate(classifier, X, y, experiment_name, split_name):
    predictions = classifier.predict(X)

    metrics = {
        "experiment": experiment_name,
        "split": split_name,
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro"),
        "sarcastic_f1": f1_score(y, predictions, pos_label=1),
    }

    return metrics, predictions


def make_classifier():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=0.5,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main():
    print("Loading data...")
    df = contrast_script.load_and_prepare_data()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Encoding context...")
    train_context_emb = encode_texts(embedding_model, train_df["context"].tolist())
    test_context_emb = encode_texts(embedding_model, test_df["context"].tolist())

    print("Encoding comment...")
    train_comment_emb = encode_texts(embedding_model, train_df["comment"].tolist())
    test_comment_emb = encode_texts(embedding_model, test_df["comment"].tolist())

    print("Encoding concatenated context + comment...")
    train_concat_texts = (
        train_df["context"].astype(str) + " [SEP] " + train_df["comment"].astype(str)
    ).tolist()

    test_concat_texts = (
        test_df["context"].astype(str) + " [SEP] " + test_df["comment"].astype(str)
    ).tolist()

    train_concat_emb = encode_texts(embedding_model, train_concat_texts)
    test_concat_emb = encode_texts(embedding_model, test_concat_texts)

    experiments = {
        "context_only": (
            train_context_emb,
            test_context_emb,
        ),
        "comment_only": (
            train_comment_emb,
            test_comment_emb,
        ),
        "concatenated_context_comment": (
            train_concat_emb,
            test_concat_emb,
        ),
        "dual_embeddings_context_comment": (
            np.concatenate([train_context_emb, train_comment_emb], axis=1),
            np.concatenate([test_context_emb, test_comment_emb], axis=1),
        ),
    }

    all_metrics = []
    predictions_df = test_df.copy()

    with open("reports/ablation_embedding_inputs_summary.txt", "w", encoding="utf-8") as f:
        f.write("Embedding Input Ablation Study\n")
        f.write("=" * 50 + "\n\n")

        f.write("Goal:\n")
        f.write(
            "Compare different ways of representing the input: context only, comment only, "
            "concatenated text, and separate context/comment embeddings.\n\n"
        )

        f.write("Embedding model:\n")
        f.write(f"{EMBEDDING_MODEL}\n\n")

        for experiment_name, (X_train, X_test) in experiments.items():
            print(f"Training {experiment_name}...")

            classifier = make_classifier()
            classifier.fit(X_train, y_train)

            train_metrics, train_predictions = evaluate(
                classifier,
                X_train,
                y_train,
                experiment_name,
                "train",
            )

            test_metrics, test_predictions = evaluate(
                classifier,
                X_test,
                y_test,
                experiment_name,
                "test",
            )

            all_metrics.extend([train_metrics, test_metrics])
            predictions_df[f"prediction_{experiment_name}"] = test_predictions

            f.write(f"Experiment: {experiment_name}\n")
            f.write("-" * 50 + "\n")
            f.write("Train metrics:\n")
            f.write(str(train_metrics))
            f.write("\n\n")
            f.write("Test metrics:\n")
            f.write(str(test_metrics))
            f.write("\n\n")
            f.write("Test classification report:\n")
            f.write(
                classification_report(
                    y_test,
                    test_predictions,
                    target_names=["not_sarcastic", "sarcastic"],
                    digits=4,
                )
            )
            f.write("\nTest confusion matrix:\n")
            f.write(str(confusion_matrix(y_test, test_predictions)))
            f.write("\n\n")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("reports/ablation_embedding_inputs_metrics.csv", index=False)
    predictions_df.to_csv("reports/ablation_embedding_inputs_predictions.csv", index=False)

    print("Done.")
    print(metrics_df)
    print()
    print("Files created:")
    print("reports/ablation_embedding_inputs_metrics.csv")
    print("reports/ablation_embedding_inputs_predictions.csv")
    print("reports/ablation_embedding_inputs_summary.txt")


if __name__ == "__main__":
    main()
