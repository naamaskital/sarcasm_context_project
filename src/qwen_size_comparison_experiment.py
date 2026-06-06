import os
import re
import gc
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


INPUT_PATH = "data/reddit_sarcasm_context_sample.csv"

MODEL_NAMES = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

RANDOM_STATE = 42
SAMPLE_PER_CLASS = 100
FEW_SHOT_PER_CLASS = 2

os.makedirs("reports", exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def label_to_text(label):
    return "sarcastic" if int(label) == 1 else "not_sarcastic"


def parse_prediction(text):
    text = text.strip().lower()

    # Important: check not_sarcastic first, because it contains "sarcastic".
    if "not_sarcastic" in text or "not sarcastic" in text:
        return 0

    if re.search(r"\bsarcastic\b", text):
        return 1

    return -1


def build_few_shot_examples(few_shot_df, with_context):
    examples = []

    for _, row in few_shot_df.iterrows():
        if with_context:
            example_text = f"""
Previous Reddit message:
{row["context"]}

Reply:
{row["comment"]}

Correct label:
{label_to_text(row["label"])}
""".strip()
        else:
            example_text = f"""
Reply:
{row["comment"]}

Correct label:
{label_to_text(row["label"])}
""".strip()

        examples.append(example_text)

    return "\n\n".join(examples)


def build_prompt(context, comment, few_shot_text, with_context):
    if with_context:
        task_text = f"""
Now classify this new example.

Previous Reddit message:
{context}

Reply:
{comment}

Answer with exactly one label:
sarcastic
or
not_sarcastic
""".strip()
    else:
        task_text = f"""
Now classify this new example.

Reply:
{comment}

Answer with exactly one label:
sarcastic
or
not_sarcastic
""".strip()

    user_content = f"""
You will see Reddit replies.
Your task is to classify whether the reply is sarcastic.

Labels:
sarcastic
not_sarcastic

Examples:
{few_shot_text}

{task_text}
""".strip()

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
            "content": user_content
        }
    ]


def predict_one(model, tokenizer, context, comment, few_shot_text, with_context):
    messages = build_prompt(
        context=context,
        comment=comment,
        few_shot_text=few_shot_text,
        with_context=with_context
    )

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return parse_prediction(answer), answer


def evaluate_model(model_name, sample_df, few_shot_df):
    print("\n" + "=" * 70)
    print(f"Loading model: {model_name}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    model.eval()

    results = []
    comparison_df = sample_df[["context", "comment", "label"]].copy()
    reports = {}

    for setting_name, with_context in [
        ("comment_only", False),
        ("context_plus_comment", True),
    ]:
        print(f"\nEvaluating {model_name} | {setting_name}")

        few_shot_text = build_few_shot_examples(
            few_shot_df=few_shot_df,
            with_context=with_context
        )

        predictions = []
        raw_answers = []

        for idx, row in sample_df.iterrows():
            print(f"{setting_name}: {len(predictions) + 1}/{len(sample_df)}")

            pred, raw_answer = predict_one(
                model=model,
                tokenizer=tokenizer,
                context=str(row["context"]),
                comment=str(row["comment"]),
                few_shot_text=few_shot_text,
                with_context=with_context
            )

            predictions.append(pred)
            raw_answers.append(raw_answer)

        y_true = sample_df["label"].astype(int).tolist()

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
                target_names=["not_sarcastic", "sarcastic"],
                zero_division=0
            )

        result = {
            "method": "qwen_few_shot",
            "model": model_name,
            "setting": setting_name,
            "num_examples": len(sample_df),
            "valid_predictions": len(valid_predictions),
            "unknown_predictions": unknown_count,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "sarcastic_f1": sarcastic_f1,
        }

        results.append(result)
        reports[f"{model_name} | {setting_name}"] = report

        safe_model_name = model_name.replace("/", "_").replace(".", "_")
        comparison_df[f"prediction_{safe_model_name}_{setting_name}"] = predictions
        comparison_df[f"raw_answer_{safe_model_name}_{setting_name}"] = raw_answers

    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, comparison_df, reports


def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_PATH)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    few_shot_df = (
        train_df.groupby("label", group_keys=False)
        .sample(n=FEW_SHOT_PER_CLASS, random_state=RANDOM_STATE)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    sample_df = (
        test_df.groupby("label", group_keys=False)
        .sample(n=SAMPLE_PER_CLASS, random_state=RANDOM_STATE)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    print("\nFew-shot examples:")
    print(few_shot_df["label"].value_counts().rename(index={
        0: "not_sarcastic",
        1: "sarcastic"
    }))

    print("\nEvaluation sample:")
    print(sample_df["label"].value_counts().rename(index={
        0: "not_sarcastic",
        1: "sarcastic"
    }))

    all_results = []
    final_comparison_df = sample_df[["context", "comment", "label"]].copy()
    all_reports = {}

    for model_name in MODEL_NAMES:
        results, comparison_df, reports = evaluate_model(
            model_name=model_name,
            sample_df=sample_df,
            few_shot_df=few_shot_df
        )

        all_results.extend(results)

        for column in comparison_df.columns:
            if column not in final_comparison_df.columns:
                final_comparison_df[column] = comparison_df[column]

        all_reports.update(reports)

    metrics_df = pd.DataFrame(all_results)

    metrics_df.to_csv("reports/qwen_size_comparison_metrics.csv", index=False)
    final_comparison_df.to_csv("reports/qwen_size_comparison_predictions.csv", index=False)

    with open("reports/qwen_size_comparison_summary.txt", "w", encoding="utf-8") as f:
        f.write("Qwen size comparison experiment\n")
        f.write("=" * 50 + "\n\n")

        f.write("Goal:\n")
        f.write(
            "Compare two sizes of a GPT/decoder-style model, Qwen 0.5B and Qwen 1.5B, "
            "using the same few-shot prompting setup.\n\n"
        )

        f.write(f"Few-shot examples per class: {FEW_SHOT_PER_CLASS}\n")
        f.write(f"Evaluation examples per class: {SAMPLE_PER_CLASS}\n")
        f.write(f"Total evaluation examples: {len(sample_df)}\n\n")

        f.write("Metrics:\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")

        for name, report in all_reports.items():
            f.write(f"Classification report - {name}:\n")
            f.write(report)
            f.write("\n\n")

        best = metrics_df.sort_values("macro_f1", ascending=False).iloc[0]
        f.write("Best setting by Macro-F1:\n")
        f.write(f"{best['model']} | {best['setting']} | Macro-F1 = {best['macro_f1']:.4f}\n")

    print("\nDone.")
    print(metrics_df)

    print("\nFiles created:")
    print("reports/qwen_size_comparison_metrics.csv")
    print("reports/qwen_size_comparison_predictions.csv")
    print("reports/qwen_size_comparison_summary.txt")


if __name__ == "__main__":
    main()
