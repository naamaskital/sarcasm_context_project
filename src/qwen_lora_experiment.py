import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score
import os

os.makedirs("reports_large_data", exist_ok=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'macro_f1': macro_f1}

def main():
    print("Loading data...")
    # טוענים מדגם - אפשר לשנות ל-4000 אם רוצים
    df = pd.read_csv("data/reddit_sarcasm_context_sample.csv")
    if len(df) > 4000:
        df = df.sample(n=4000, random_state=42)
        
    # בניית הקלט: הקשר + תגובה
    df['text'] = "Context: " + df['context'].astype(str) + "\nComment: " + df['comment'].astype(str)
    
    dataset = Dataset.from_pandas(df[['text', 'label']])
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading Tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize, batched=True)

    print("Loading Base Model for Sequence Classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=2,
        device_map="auto" # שולח אוטומטית ל-GPU
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Applying LoRA Config...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="reports_large_data/qwen_lora_checkpoints",
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    print("Starting LoRA Fine-Tuning...")
    trainer.train()

    print("\nEvaluating best model on test set...")
    results = trainer.evaluate()
    print(results)
    
    # שמירת התוצאות
    with open("reports_large_data/qwen_lora_summary.txt", "w", encoding="utf-8") as f:
        f.write("Qwen 0.5B LoRA Fine-Tuning Results\n")
        f.write("="*40 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()