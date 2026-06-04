import os
import re
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


INPUT_PATH = "data/reddit_sarcasm_context_sample.csv"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

RANDOM_STATE = 42
SAMPLE_PER_CLASS = 10

os.makedirs("reports", exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_prompt(context, comment, with_context):
    if with_context:
        user_text = f"""
Previous Reddit message:
{context}

Reply:
{comment}

Question:
Is the reply sarcastic?

Answer with exactly one label:
sarcastic
or
not_sarcastic
"""
    else:
        user_text = f"""
Reply:
{comment}

Question:
Is the reply sarcastic?

Answer with exactly one label:
sarcastic
or
not_sarcastic
"""

    return [
        {
            "role": "system",
            "content": (
                "You are a strict binary classifier for sarcasm detection. "
                "Answer only with sarcastic or not_sarcastic."
            )
        },
        {
            "role": "user",
            "content": user_text.strip()
        }
    ]


def parse_prediction(text):
    text = text.strip().lower()

    # Important: check not_sarcastic first because it contains the word sarcastic.
    if "not_sarcastic" in text or "not sarcastic" in text:
        return 0

    if re.search(r"\bsarcastic\b", text):
        return 1

    return -1


def predict_one(model, tokenizer, device, context, comment, with_context):
    messages = build_prompt(context, comment, with_context)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return parse_prediction(answer), answer


def evaluate_setting(model, tokenizer, device, df, setting_name, with_context):
    predictions = []
    raw_answers = []

    total = len(df)

    for i, row in df.iterrows():
        print(f"{setting_name}: {len(predictions) + 1}/{total}")

        pred, raw_answer = predict_one(
            model=model,
            tokenizer=tokenizer,
            device=device,
            context=str(row["context"]),
            comment=str(row["comment"]),
            with_context=with_context
        )

        predictions.append(pred)
        raw_answers.append(raw_answer)

    y_true = df["label"].astype(int).tolist()

    valid_indices = [i for i, p in enumerate(predictions) if p in [0, 1]]

    valid_y_true = [y_true[i] for i in valid_indices]
    valid_predictions = [predictions[i] for i in valid_indices]

    unknown_count = len(predictions) - len(valid_predictions)

    if len(valid_predictions) == 0:
        accuracy = 0.0
        macro_f1 = 0.0
        sarcastic_f1 = 0.0
        report = "No valid predictions."
    else:
        accuracy = accuracy_score(valid_y_true, valid_predictions)
        macro_f1 = f1_score(valid_y_true, valid_predictions, average="macro")
        sarcastic_f1 = f1_score(valid_y_true, valid_predictions, pos_label=1)
        report = classification_report(
            valid_y_true,
            valid_predictions,
            target_names=["not_sarcastic", "sarcastic"]
        )

    result = {
        "method": "qwen_zero_shot",
        "model": MODEL_NAME,
        "setting": setting_name,
        "num_examples": len(df),
        "valid_predictions": len(valid_predictions),
        "unknown_predictions": unknown_count,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "sarcastic_f1": sarcastic_f1,
    }

    return result, predictions, raw_answers, report


def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_PATH)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    # Balanced small sample from the test set
    sample_df = (
        test_df.groupby("label", group_keys=False)
        .sample(n=SAMPLE_PER_CLASS, random_state=RANDOM_STATE)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    print("Sample size:")
    print(sample_df["label"].value_counts().rename(index={
        0: "not_sarcastic",
        1: "sarcastic"
    }))

    print("\nLoading Qwen model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype
    )
    model.to(device)
    model.eval()

    results = []
    reports = {}

    comparison_df = sample_df[["context", "comment", "label"]].copy()

    for setting_name, with_context in [
        ("comment_only", False),
        ("context_plus_comment", True),
    ]:
        result, predictions, raw_answers, report = evaluate_setting(
            model=model,
            tokenizer=tokenizer,
            device=device,
            df=sample_df,
            setting_name=setting_name,
            with_context=with_context
        )

        results.append(result)
        reports[setting_name] = report

        comparison_df[f"prediction_{setting_name}"] = predictions
        comparison_df[f"raw_answer_{setting_name}"] = raw_answers

    metrics_df = pd.DataFrame(results)

    metrics_df.to_csv("reports/qwen_zero_shot_metrics.csv", index=False)
    comparison_df.to_csv("reports/qwen_zero_shot_predictions.csv", index=False)

    with open("reports/qwen_zero_shot_summary.txt", "w", encoding="utf-8") as f:
        f.write("Qwen zero-shot experiment\n")
        f.write("=" * 45 + "\n\n")

        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Sample per class: {SAMPLE_PER_CLASS}\n")
        f.write(f"Total examples: {len(sample_df)}\n\n")

        f.write("Goal:\n")
        f.write(
            "Compare comment only vs context + comment using a decoder/GPT-style model "
            "without fine-tuning, only prompting.\n\n"
        )

        f.write("Metrics:\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")

        for setting_name, report in reports.items():
            f.write(f"Classification report - {setting_name}:\n")
            f.write(report)
            f.write("\n\n")

    print("\nDone.")
    print(metrics_df)

    print("\nFiles created:")
    print("reports/qwen_zero_shot_metrics.csv")
    print("reports/qwen_zero_shot_predictions.csv")
    print("reports/qwen_zero_shot_summary.txt")


if __name__ == "__main__":
    main()
