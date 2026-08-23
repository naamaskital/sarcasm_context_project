# Sarcasm Detection with Conversational Context

Final project for **Advanced Models of Language Understanding**.

## Research question

**Does the previous Reddit message improve sarcasm detection in a reply, and do models use the actual conversational relation or only topical / semantic cues?**

The project is built around three controlled input conditions:

1. **Comment Only** – the target Reddit reply.
2. **Context Only** – the previous Reddit message.
3. **Context + Comment** – the real conversational pair.

Additional ablations replace the true context with incorrect context to test whether a model really depends on the conversational relation.

## Dataset

The main experiments use the SARC Reddit sarcasm corpus. After removing examples with missing or empty reply/context text, the fixed IID split is:

| Split | Examples |
|---|---:|
| Train | 808,616 |
| Validation | 101,077 |
| Test | 101,078 |

Seed: `42`.

## Experimental progression

The project deliberately progresses from simple lexical baselines to stronger contextual models and counterfactual tests:

- **TF-IDF + Logistic Regression** – lexical baseline.
- **Field-aware TF-IDF / selective context routing** – failure-driven follow-ups.
- **MiniLM embeddings** – frozen semantic representations.
- **Hard-context ablations** – random, same-subreddit, and semantically similar wrong contexts.
- **BERT / RoBERTa / DeBERTa cross-encoders** – joint token-level interaction.
- **FLAN-T5** – encoder-decoder comparison.
- **Qwen2.5 + LoRA** – decoder-only comparison and controlled context ablation.
- **Qwen scaling controls** – 0.5B vs 1.5B under a fixed few-shot protocol.
- **Unseen-subreddit evaluation** – community generalization.

The central design principle is not only *which model is best*, but *how and when context is actually used*.

## Canonical final-report results

The canonical compact table used by the final scientific report is stored in:

`reports/final_report_results.csv`

### Full-dataset IID results (Macro-F1)

| Model | Comment Only | Context Only | Context + Comment |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | **0.7242** | 0.5856 | 0.7045 |
| MiniLM + linear classifier | 0.6673 | 0.5917 | **0.6719** |
| BERT-base cross-encoder | 0.7783 | 0.6097 | **0.7884** |
| RoBERTa-base cross-encoder | 0.7840 | 0.6159 | **0.7955** |
| DeBERTa-v3-base cross-encoder | 0.7901 | 0.6205 | **0.8030** |
| FLAN-T5-base | 0.7561 | 0.5981 | **0.7705** |
| Qwen2.5-0.5B + LoRA | 0.7385 | 0.6010 | **0.7521** |

### Failure-driven lexical follow-ups

- Field-aware TF-IDF: **0.7297** Macro-F1.
- Selective context routing: **0.7317** Macro-F1.

### Controlled Qwen2.5-0.5B + LoRA context ablation

This is a **separate balanced controlled protocol**, not the full-dataset Qwen experiment: 3,000 train / 500 validation / 1,000 test examples, trained for 5 epochs.

| Input | Macro-F1 |
|---|---:|
| Context Only | 0.5613 |
| Comment Only | 0.6830 |
| True Context + Comment | **0.6850** |
| Random Context + Comment | 0.6528 |

### Hard context ablation (MiniLM)

| Context condition | Macro-F1 |
|---|---:|
| True context | **0.6540** |
| Random wrong context | 0.6375 |
| Same-subreddit wrong context | 0.6390 |
| Semantically similar wrong context | 0.6540 |

Semantically similar hard negatives are selected with normalized `all-MiniLM-L6-v2` context embeddings using cosine similarity, excluding self-matches and identical context strings. Confidence intervals are computed with 1,000 paired bootstrap resamples.

### Qwen scaling control

A fixed balanced 1,000-example test sample is evaluated with four demonstrations total (two per class).

| Model | Comment | True Context | Random Context |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 0.4350 | 0.4337 | 0.4213 |
| Qwen2.5-1.5B-Instruct | 0.4603 | 0.5249 | 0.5179 |

### Unseen-subreddit generalization

| Model | Comment Only | Context-aware |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.7099 | 0.6887 |
| MiniLM + linear classifier | 0.6578 | 0.6598 |

## Main conclusions

- **Context alone is weak** across model families.
- **The reply is the strongest individual signal** in most settings.
- **Adding context is not automatically useful**: naive lexical concatenation can hurt.
- **Representation matters**: semantic and joint token-level interaction can use context more effectively.
- **Correct context matters in some settings**, but context sensitivity depends on architecture, model size, and the type of negative context.
- **Semantic compatibility is not equivalent to exact conversational dependence**: semantically similar wrong context can perform like the true parent for the MiniLM hard-context classifier.
- **Context does not help every example**; selective use can outperform unconditional concatenation.

Overall, the value of context is **architecture-, representation-, protocol-, and example-dependent**.

## Repository structure

```text
.
├── README.md
├── RESULTS.md
├── PROJECT_FLOW.md
├── requirements.txt
├── run_final_pipeline.sh
├── src/
├── scripts/
└── reports/
    ├── final_report_results.csv
    ├── full_dataset/
    ├── hard_context_ablation/
    ├── qwen_lora_ablation/
    └── qualitative_analysis/
```

## Reproduction

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Examples:

```bash
python scripts/18_full_dataset_tfidf.py
python scripts/19_full_dataset_embeddings.py
python scripts/16_hard_context_ablation.py
python scripts/30_bert_cross_encoder.py --mode all --resume --skip-completed
```

GPU experiments require a CUDA-capable machine.

## References

- Khodak, M., Saunshi, N., & Vodrahalli, K. (2018). *A Large Self-Annotated Corpus for Sarcasm*. LREC.
- Hazarika, D. et al. (2018). *CASCADE: Contextual Sarcasm Detection in Online Discussion Forums*. COLING.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
- Hu, E. J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR.
