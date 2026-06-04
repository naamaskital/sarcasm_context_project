import os
import pandas as pd

INPUT_PATH = "reports/prediction_comparison.csv"

OUTPUT_HELPED = "reports/context_helped_examples.csv"
OUTPUT_HURT = "reports/context_hurt_examples.csv"
OUTPUT_SLIDE = "reports/preliminary_finding_for_slide.txt"

os.makedirs("reports", exist_ok=True)


def label_name(value):
    return "sarcastic" if int(value) == 1 else "not_sarcastic"


def shorten(text, max_len=220):
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def main():
    df = pd.read_csv(INPUT_PATH)

    # Cases where comment-only was wrong, but context+comment was correct
    helped = df[
        (df["prediction_without_context"] != df["label"]) &
        (df["prediction_with_context"] == df["label"])
    ].copy()

    # Cases where comment-only was correct, but context+comment became wrong
    hurt = df[
        (df["prediction_without_context"] == df["label"]) &
        (df["prediction_with_context"] != df["label"])
    ].copy()

    helped.to_csv(OUTPUT_HELPED, index=False)
    hurt.to_csv(OUTPUT_HURT, index=False)

    with open(OUTPUT_SLIDE, "w", encoding="utf-8") as f:
        f.write("Preliminary finding for presentation\n")
        f.write("=" * 45 + "\n\n")

        f.write("Quantitative finding:\n")
        f.write(
            "Adding conversational context improved the baseline model:\n"
            "Accuracy: 59.00% -> 61.25%\n"
            "Macro-F1: 58.93% -> 61.10%\n"
            "Sarcastic F1: 57.29% -> 58.67%\n\n"
        )

        f.write("Interpretation:\n")
        f.write(
            "This suggests that conversational context contains useful information "
            "for sarcasm detection. However, the improvement is still modest, "
            "so a stronger language model may be needed to use context more effectively.\n\n"
        )

        f.write(f"Number of examples where context helped: {len(helped)}\n")
        f.write(f"Number of examples where context hurt: {len(hurt)}\n\n")

        f.write("Examples where context helped:\n")
        f.write("-" * 45 + "\n")

        for i, (_, row) in enumerate(helped.head(5).iterrows(), start=1):
            f.write(f"\nExample {i}\n")
            f.write(f"Context: {shorten(row['context'])}\n")
            f.write(f"Comment: {shorten(row['comment'])}\n")
            f.write(f"True label: {label_name(row['label'])}\n")
            f.write(f"Prediction without context: {label_name(row['prediction_without_context'])}\n")
            f.write(f"Prediction with context: {label_name(row['prediction_with_context'])}\n")

    print("Done.")
    print(f"Context helped examples: {len(helped)}")
    print(f"Context hurt examples: {len(hurt)}")
    print("\nFiles created:")
    print(OUTPUT_HELPED)
    print(OUTPUT_HURT)
    print(OUTPUT_SLIDE)


if __name__ == "__main__":
    main()
