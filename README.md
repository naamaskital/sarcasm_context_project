# Sarcasm Detection with Conversational Context

Final project for **Advanced Models of Language Understanding**.

## Research question

**Does the previous Reddit message provide useful information for detecting sarcasm in a reply?**

The project compares four controlled input conditions:

1. **Comment only** - the target Reddit reply.
2. **Context only** - the previous message.
3. **True context + comment** - the real conversational pair.
4. **Random context + comment** - the same reply paired with an unrelated context.

The random-context condition is the key ablation: it separates a benefit from *meaningful conversational context* from a benefit caused only by adding more text.

## Dataset

The text fields are:

- `comment` - target Reddit reply
- `parent_comment` / `context` - previous Reddit message
- `label` - binary sarcasm label

Two data settings are kept separate in this repository:

- `data_backup/reddit_sarcasm_context_sample.csv` is a small balanced **2,000-example sample** bundled with the repository for quick sanity checks.
- The final Qwen + LoRA experiments used the larger Hugging Face dataset `marcbishara/sarcasm-on-reddit`. The logged final split was fully balanced: **3,000 train, 500 validation, 1,000 test**.

This distinction matters: the bundled-sample sanity run should not be compared numerically with the larger final experiments as if they were the same split.

## Methods

### 1. TF-IDF + Logistic Regression

A classical lexical baseline using unigram and bigram TF-IDF features and balanced Logistic Regression. It tests how much sarcasm can be predicted from recurring lexical cues without a neural language model.

### 2. Sentence Transformer embeddings

`all-MiniLM-L6-v2` is used as a frozen semantic encoder. Experiments include joint text representations and separate context/comment embeddings followed by a classifier.

### 3. Qwen2.5-0.5B-Instruct + LoRA

The final language-model experiment treats Qwen as a binary sequence classifier and adapts it using LoRA rather than full fine-tuning.

Final configuration documented by the saved training run:

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.1
- Target modules: `q_proj`, `v_proj`
- Learning rate: `2e-4`
- Batch size: 8
- Epochs: 5
- Trainable parameters in the logged run: about 542K of 494.6M (~0.11%)

`src/qwen_lora_context_ablation.py` reconstructs this final experimental configuration and all four input conditions. The exact saved metrics from the completed run are stored in `reports/qwen_lora_ablation/qwen_lora_context_ablation_summary.csv`.

## Final results

The main verified/report results are collected in `reports/final_results.csv`.

| Model | Input | Accuracy | Macro-F1 | Sarcastic F1 |
|---|---|---:|---:|---:|
| TF-IDF + Logistic Regression | context only | 0.5425 | 0.5425 | 0.5395 |
| TF-IDF + Logistic Regression | comment only | 0.6430 | 0.6426 | 0.6301 |
| TF-IDF + Logistic Regression | true context + comment | 0.6500 | 0.6499 | 0.6436 |
| Sentence Transformer, separate embeddings | true context + comment | 0.6335 | 0.6334 | 0.6380 |
| Qwen2.5-0.5B + LoRA | context only | 0.5660 | 0.5613 | 0.6069 |
| Qwen2.5-0.5B + LoRA | comment only | 0.6830 | 0.6830 | 0.6801 |
| Qwen2.5-0.5B + LoRA | true context + comment | **0.6850** | **0.6850** | **0.6859** |
| Qwen2.5-0.5B + LoRA | random context + comment | 0.6530 | 0.6528 | 0.6615 |

### Context ablation

The most important Qwen comparison is:

- true context + comment: Macro-F1 **0.6850**
- random context + comment: Macro-F1 **0.6528**

The true-context model is only slightly better than comment-only, but replacing the true context with unrelated text causes a clear drop. This supports a cautious conclusion: **the reply contains most of the predictive signal, while the correct context provides additional information that the fine-tuned Qwen model can use.**

The classical and embedding experiments also show that simply adding context does not guarantee improvement. In particular, some sentence-embedding variants were relatively insensitive to whether the context was correct, suggesting that representation design matters.

### Hard context ablation

A stronger controlled experiment was added to test whether the embedding-based classifier uses the exact conversational context or only broad semantic/topic compatibility. The classifier was trained once on true context + comment using a balanced 20,000-example sample, then evaluated on the same 4,000-example test set while only the context was replaced.

| Test context | Accuracy | Macro-F1 | 95% paired bootstrap delta vs. true |
|---|---:|---:|---:|
| true context | 0.6540 | 0.6540 | - |
| random wrong context | 0.6375 | 0.6375 | [0.0035, 0.0293] |
| same-subreddit wrong context | 0.6390 | 0.6390 | [0.0020, 0.0290] |
| semantically similar wrong context | 0.6540 | 0.6540 | [-0.0113, 0.0118] |

Same-subreddit hard negatives were available for **84.7%** of the test set. The mean cosine similarity of the semantic hard negatives was **0.4451**.

The result is intentionally nuanced: replacing the real context with random or same-subreddit context causes a statistically consistent drop in Macro-F1, but a semantically similar incorrect context performs essentially the same as the true context. This suggests that the frozen embedding classifier benefits from **semantic compatibility** between context and reply, yet does not reliably identify the exact conversational dependency. This limitation motivates stronger interaction models such as cross-encoders or fine-tuned LLMs.

Detailed results are stored under `reports/hard_context_ablation/`, and the experiment is implemented in `scripts/16_hard_context_ablation.py`.

## Reproducing the project

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Quick bundled-sample baseline

```bash
python src/context_control_experiment.py
```

This automatically uses `data_backup/reddit_sarcasm_context_sample.csv` when no `data/` copy exists and writes results to `reports/sample_baseline/`.

### Sentence Transformer experiment

```bash
python src/sentence_transformer_experiment.py
```

Additional embedding analyses are under `scripts/`.

### Hard context ablation

```bash
python scripts/16_hard_context_ablation.py
```

This experiment trains once on true context + comment and replaces only the test context with random, same-subreddit, and semantically similar hard negatives. It also reports paired 95% bootstrap confidence intervals.

### Final Qwen + LoRA ablation

```bash
python src/qwen_lora_context_ablation.py
```

This experiment downloads the Hugging Face dataset and Qwen checkpoint. A CUDA-capable GPU is strongly recommended. Outputs are written to `reports/qwen_lora_ablation/` and checkpoints to `models/qwen_lora_ablation/`.

### Exploratory Qwen experiments

The repository also keeps earlier zero-shot, few-shot, model-size, and preliminary LoRA scripts for the experimental record. They are useful for showing the development process, but the final Qwen result reported above is the five-epoch LoRA context-ablation experiment.

## Repository structure

```text
.
├── data_backup/
│   └── reddit_sarcasm_context_sample.csv
├── reports/
│   ├── final_results.csv
│   ├── hard_context_ablation/
│   │   ├── hard_context_ablation_metrics.csv
│   │   └── hard_context_ablation_summary.txt
│   └── qwen_lora_ablation/
│       └── qwen_lora_context_ablation_summary.csv
├── reports_backup/                 # earlier experiment outputs
├── scripts/
│   ├── 13_train_contrast_features.py
│   ├── 14_train_dual_embeddings.py
│   ├── 15_ablation_embedding_inputs.py
│   └── 16_hard_context_ablation.py
├── src/
│   ├── context_control_experiment.py
│   ├── sentence_transformer_experiment.py
│   ├── qwen_zero_shot_experiment.py
│   ├── qwen_size_comparison_experiment.py
│   ├── qwen_lora_experiment.py     # earlier LoRA version
│   └── qwen_lora_context_ablation.py
├── requirements.txt
└── README.md
```

## Main conclusions

- **Context alone is weak**: the previous message is not a reliable sarcasm label by itself.
- **The reply is the main source of signal**: comment-only is already substantially stronger.
- **True context can add useful information**, but the gain over comment-only is modest.
- **Random-context ablation is essential**: for Qwen, unrelated context reduces performance substantially compared with the true conversational context.
- **Hard-negative analysis reveals a limitation**: the embedding classifier distinguishes true context from random and same-community context, but not from semantically similar incorrect context.
- **A larger or more semantic model is not automatically better**: representation choices affect whether a model actually uses conversational relationships rather than topical similarity.

## References

- Khodak, M., Saunshi, N., & Vodrahalli, K. (2018). *A Large Self-Annotated Corpus for Sarcasm*. LREC.
- Hazarika, D., Poria, S., Gorantla, S., Cambria, E., Zimmermann, R., & Mihalcea, R. (2018). *CASCADE: Contextual Sarcasm Detection in Online Discussion Forums*. COLING.
- Hu, E. J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR.
