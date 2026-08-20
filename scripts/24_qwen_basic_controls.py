from pathlib import Path
import argparse
import gc
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.full_dataset_utils import load_or_create_splits

SEED = 42
MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]
DEFAULT_EXAMPLES_PER_CLASS = 500
FEW_SHOT_PER_CLASS = 2
MAX_LENGTH = 1024
REPORT_DIR = ROOT / "reports" / "full_dataset" / "qwen_basic_controls"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Qwen basic controls: model-size comparison reframed as context-utilization, "
            "plus prompt-formatting ablation on a fixed balanced held-out sample."
        )
    )
    parser.add_argument("--examples-per-class", type=int, default=DEFAULT_EXAMPLES_PER_CLASS)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen basic control experiments.")


def label_text(label):
    return "1" if int(label) == 1 else "0"


def deranged_contexts(df, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(df)
    base = np.arange(n)
    perm = rng.permutation(n)
    while np.any(perm == base):
        perm = rng.permutation(n)
    return df.iloc[perm]["context"].to_numpy()


def context_value(row, input_mode):
    if input_mode == "context_plus_comment":
        return row["context"]
    if input_mode == "random_context_plus_comment":
        return row["random_context"]
    raise ValueError(input_mode)


def format_example(row, input_mode, prompt_format, include_label=True):
    if input_mode == "comment_only":
        body = f"Reply:\n{row['comment']}"
    elif input_mode in ["context_plus_comment", "random_context_plus_comment"]:
        context = context_value(row, input_mode)
        if prompt_format == "structured":
            body = f"Previous Reddit message:\n{context}\n\nReply:\n{row['comment']}"
        elif prompt_format == "plain_concat":
            body = f"{context}\n{row['comment']}"
        else:
            raise ValueError(prompt_format)
    else:
        raise ValueError(input_mode)

    if include_label:
        return body + f"\n\nLabel: {label_text(row['label'])}"
    return body


def build_messages(row, few_shot_df, input_mode, prompt_format):
    # Demonstrations are always internally coherent. Only the target test context is perturbed.
    demo_mode = "comment_only" if input_mode == "comment_only" else "context_plus_comment"
    demos = [
        format_example(demo, demo_mode, prompt_format, include_label=True)
        for _, demo in few_shot_df.iterrows()
    ]

    target = format_example(row, input_mode, prompt_format, include_label=False)
    user_text = (
        "Classify whether the Reddit reply is sarcastic.\n"
        "Output exactly one digit: 1 = sarcastic, 0 = not sarcastic.\n\n"
        "Examples:\n\n"
        + "\n\n---\n\n".join(demos)
        + "\n\n---\n\nNew example:\n"
        + target
        + "\n\nLabel:"
    )

    return [
        {"role": "system", "content": "You are a strict binary sarcasm classifier. Answer only with 0 or 1."},
        {"role": "user", "content": user_text},
    ]


def parse_label(text):
    match = re.search(r"[01]", text.strip())
    return int(match.group()) if match else -1


def make_balanced_sample(df, examples_per_class, seed):
    counts = df["label"].value_counts()
    if any(counts.get(label, 0) < examples_per_class for label in [0, 1]):
        raise ValueError(f"Not enough examples. Counts: {counts.to_dict()}")
    out = (
        df.groupby("label", group_keys=False)
        .sample(n=examples_per_class, random_state=seed)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    out["random_context"] = deranged_contexts(out, seed + 99)
    return out


def make_few_shot(train_df):
    return (
        train_df.groupby("label", group_keys=False)
        .sample(n=FEW_SHOT_PER_CLASS, random_state=SEED)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Both Qwen models used here fit comfortably on the 8 GB project GPU in FP16.
    # Explicit placement avoids an Accelerate/device_map="auto" dispatch hang seen on
    # the RTX 2070 + PyTorch cu118 environment used for the final run.
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to("cuda")
    model.eval()
    print("Model device:", next(model.parameters()).device)
    return model, tokenizer


def predict_batches(model, tokenizer, eval_df, few_shot_df, input_mode, prompt_format, batch_size):
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(row, few_shot_df, input_mode, prompt_format),
            tokenize=False,
            add_generation_prompt=True,
        )
        for _, row in eval_df.iterrows()
    ]

    predictions, raw_answers = [], []
    device = next(model.parameters()).device
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        encoded = tokenizer(
            prompts[start:end], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_len = encoded["input_ids"].shape[1]
        answers = tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True)
        raw_answers.extend(answers)
        predictions.extend(parse_label(answer) for answer in answers)
        print(f"  {end:,}/{len(prompts):,}")
    return np.asarray(predictions, dtype=np.int64), raw_answers


def metrics(y, pred):
    valid = np.isin(pred, [0, 1])
    if valid.any():
        valid_acc = float(accuracy_score(y[valid], pred[valid]))
        macro = float(f1_score(y[valid], pred[valid], average="macro"))
        sarcastic = float(f1_score(y[valid], pred[valid], pos_label=1))
    else:
        valid_acc = macro = sarcastic = 0.0
    return {
        "accuracy_all": float((valid & (pred == y)).mean()),
        "accuracy_valid": valid_acc,
        "macro_f1_valid": macro,
        "sarcastic_f1_valid": sarcastic,
        "valid_rate": float(valid.mean()),
        "unknown_predictions": int((~valid).sum()),
    }


def experiment_settings(model_id):
    settings = [
        ("comment_only", "structured", "context_utilization_scaling"),
        ("context_plus_comment", "structured", "context_utilization_scaling"),
        ("random_context_plus_comment", "structured", "context_utilization_scaling"),
    ]
    if model_id.endswith("0.5B-Instruct"):
        settings.append(("context_plus_comment", "plain_concat", "prompt_format_ablation"))
    return settings


def utilization_table(results):
    rows = []
    scaling = results[results["purpose"] == "context_utilization_scaling"]
    for model_id, group in scaling.groupby("model"):
        by_input = group.set_index("input")
        required = ["comment_only", "context_plus_comment", "random_context_plus_comment"]
        if not all(name in by_input.index for name in required):
            continue
        f_comment = float(by_input.loc["comment_only", "macro_f1_valid"])
        f_true = float(by_input.loc["context_plus_comment", "macro_f1_valid"])
        f_random = float(by_input.loc["random_context_plus_comment", "macro_f1_valid"])
        rows.append({
            "model": model_id,
            "comment_macro_f1": f_comment,
            "true_context_macro_f1": f_true,
            "random_context_macro_f1": f_random,
            "context_gain": f_true - f_comment,
            "context_sensitivity": f_true - f_random,
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    require_cuda()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, _, test_df = load_or_create_splits()
    eval_df = make_balanced_sample(test_df, args.examples_per_class, SEED)
    few_shot_df = make_few_shot(train_df)
    y = eval_df["label"].to_numpy(dtype=np.int64)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Evaluation examples:", len(eval_df))

    rows = []
    prediction_table = eval_df[["subreddit", "context", "random_context", "comment", "label"]].copy()

    for model_id in MODELS:
        print("\n" + "=" * 78)
        print("Loading:", model_id)
        model, tokenizer = load_model(model_id)
        safe_model = model_id.split("/")[-1].replace(".", "_")

        for input_mode, prompt_format, purpose in experiment_settings(model_id):
            print(f"\n{model_id} | {input_mode} | {prompt_format}")
            pred, raw = predict_batches(
                model, tokenizer, eval_df, few_shot_df, input_mode, prompt_format, args.batch_size
            )
            row = {
                "purpose": purpose,
                "model": model_id,
                "input": input_mode,
                "prompt_format": prompt_format,
                "n_examples": len(eval_df),
                **metrics(y, pred),
            }
            rows.append(row)
            print(row)
            key = f"{safe_model}__{input_mode}__{prompt_format}"
            prediction_table[f"prediction__{key}"] = pred
            prediction_table[f"raw_answer__{key}"] = raw

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    utilization = utilization_table(results)
    results.to_csv(REPORT_DIR / "qwen_basic_controls_metrics.csv", index=False)
    utilization.to_csv(REPORT_DIR / "qwen_context_utilization_by_scale.csv", index=False)
    prediction_table.to_parquet(REPORT_DIR / "qwen_basic_controls_predictions.parquet", index=False)

    with open(REPORT_DIR / "qwen_basic_controls_summary.txt", "w", encoding="utf-8") as f:
        f.write("Qwen basic controls and context-utilization scaling\n")
        f.write("=" * 70 + "\n\n")
        f.write(results.to_string(index=False))
        f.write("\n\nCONTEXT UTILIZATION BY SCALE\n")
        f.write(utilization.to_string(index=False))

    print("\nFINAL BASIC CONTROLS")
    print(results.to_string(index=False))
    print("\nCONTEXT UTILIZATION BY SCALE")
    print(utilization.to_string(index=False))
    print("Saved to:", REPORT_DIR)


if __name__ == "__main__":
    main()
