import importlib.util
import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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


def build_dual_embedding_features(model, contexts, comments):
    """
    Build two separate embeddings:
    1. context embedding
    2. comment embedding

    No explicit contrast formula is added.
    The classifier has to learn the relation by itself.
    """
    context_emb = model.encode(
        contexts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    comment_emb = model.encode(
        comments,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return np.concatenate([context_emb, comment_emb], axis=1)


def evaluate(model, X, y, model_name, split_name):
    predictions = model.predict(X)

    return {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro"),
        "sarcastic_f1": f1_score(y, predictions, pos_label=1),
    }, predictions


def main():
    print("Loading data...")
    df = contrast_script.load_and_prepare_data()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Building train features...")
    X_train = build_dual_embedding_features(
        embedding_model,
        train_df["context"].tolist(),
        train_df["comment"].tolist(),
    )
    y_train = train_df["label"].values

    print("Building test features...")
    X_test = build_dual_embedding_features(
        embedding_model,
        test_df["context"].tolist(),
        test_df["comment"].tolist(),
    )
    y_test = test_df["label"].values

    models = {
        "dual_embeddings_logistic_regression": Pipeline(
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
        ),
        "dual_embeddings_mlp": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        alpha=0.001,
                        max_iter=500,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    all_metrics = []
    all_predictions = test_df.copy()

    with open("reports/dual_embeddings_summary.txt", "w", encoding="utf-8") as f:
        f.write("Dual Embeddings Experiment\n")
        f.write("=" * 50 + "\n\n")
        f.write("Goal:\n")
        f.write(
            "Represent context and comment separately, without adding explicit contrast formulas.\n"
        )
        f.write(
            "The classifier receives [v_context, v_comment] and learns the relation by itself.\n\n"
        )

        f.write("Embedding model:\n")
        f.write(f"{EMBEDDING_MODEL}\n\n")

        for model_name, classifier in models.items():
            print(f"Training {model_name}...")
            classifier.fit(X_train, y_train)

            train_metrics, train_predictions = evaluate(
                classifier, X_train, y_train, model_name, "train"
            )
            test_metrics, test_predictions = evaluate(
                classifier, X_test, y_test, model_name, "test"
            )

            all_metrics.extend([train_metrics, test_metrics])
            all_predictions[f"prediction_{model_name}"] = test_predictions

            f.write(f"Model: {model_name}\n")
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
    metrics_df.to_csv("reports/dual_embeddings_metrics.csv", index=False)
    all_predictions.to_csv("reports/dual_embeddings_predictions.csv", index=False)

    print("Done.")
    print(metrics_df)
    print()
    print("Files created:")
    print("reports/dual_embeddings_metrics.csv")
    print("reports/dual_embeddings_predictions.csv")
    print("reports/dual_embeddings_summary.txt")


if __name__ == "__main__":
    main()
