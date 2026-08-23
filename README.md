# Sarcasm Detection with Conversational Context

Final project for **Advanced Models of Language Understanding**.

## Research question

**Does the previous Reddit message improve sarcasm detection in a reply, and do models use the actual conversational relation or only topical / semantic cues?**

The project is built around three controlled input conditions:

1. **Comment Only** – the target Reddit reply.
2. **Context Only** – the previous Reddit message.
3. **Context + Comment** – the real conversational pair.

Additional ablations replace the true context with incorrect context to test whether a model really depends on the conversational relation.

---

## Dataset

The main experiments use the SARC Reddit sarcasm corpus.

Full IID split:

| Split | Examples |
|---|---:|
| Train | 808,616 |
| Validation | 101,077 |
| Test | 101,078 |

Seed: `42`.

---

## Methods

The project deliberately progresses from simple lexical models to contextual neural models:

- **TF-IDF + Logistic Regression** – lexical baseline.
- **Field-aware TF-IDF / selective context analysis** – failure-driven follow-ups to test when context helps.
- **MiniLM embeddings** – frozen semantic representations.
- **BERT cross-encoder** – joint token-level interaction between context and reply.
- **Qwen2.5 + LoRA** – decoder-only model with parameter-efficient fine-tuning.
- **Hard-context ablations** – random, same-subreddit, and semantically similar wrong contexts.
- **Unseen-subreddit evaluation** – tests community-generalization rather than only IID performance.

The central design principle is not to ask only *which model is best*, but *how and when context is actually used*.

---

## Main verified results

### Full-dataset IID experiments

| Model | Comment Only Macro-F1 | Context Only Macro-F1 | Context + Comment Macro-F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | **0.7242** | 0.5856 | 0.7045 |
| MiniLM + linear classifier | 0.6673 | 0.5917 | **0.6719** |
| BERT-base cross-encoder | 0.7783 | 0.6097 | **0.7884** |

Key observation: naive lexical context hurts TF-IDF, while semantic and especially joint cross-encoder representations can benefit from context.

### Qwen2.5-0.5B + LoRA controlled experiment

| Input | Accuracy | Macro-F1 | Sarcastic F1 |
|---|---:|---:|---:|
| Context Only | 0.5660 | 0.5613 | 0.6069 |
| Comment Only | 0.6830 | 0.6830 | 0.6801 |
| True Context + Comment | **0.6850** | **0.6850** | **0.6859** |
| Random Context + Comment | 0.6530 | 0.6528 | 0.6615 |

The reply contains most of the signal, but replacing the true context with unrelated text causes a clear degradation.

### Hard context ablation

A stronger diagnostic evaluates one context-aware MiniLM classifier while replacing only the context at test time.

| Context condition | Macro-F1 | 95% paired bootstrap delta vs. true |
|---|---:|---:|
| True context | **0.6540** | – |
| Random wrong context | 0.6375 | [0.0035, 0.0293] |
| Same-subreddit wrong context | 0.6390 | [0.0020, 0.0290] |
| Semantically similar wrong context | 0.6540 | [-0.0113, 0.0118] |

This is one of the main findings of the project: **semantic compatibility can be enough for a frozen embedding classifier, even when the context is not the true parent message.**

More result files are available under [`reports/`](reports/).

---

## Main conclusions

- **Context alone is weak** across the tested model families.
- **The reply is the strongest individual signal** for sarcasm classification.
- **Adding context is not automatically useful**: naive TF-IDF concatenation decreases performance.
- **Representation matters**: MiniLM gains slightly from separate semantic representations, while BERT gains more from joint context–reply interaction.
- **Correct context matters in some settings**, as shown by Qwen random-context degradation.
- **Semantic similarity is not the same as exact conversational dependence**: the hard-negative experiment shows that a semantically similar wrong context can behave like the true context for MiniLM.

Overall, the value of context is **architecture-, representation-, and example-dependent**.

---

## Repository structure

```text
.
├── README.md
├── RESULTS.md
├── requirements.txt
├── run_final_pipeline.sh
├── src/
│   ├── full_dataset_utils.py
│   ├── subreddit_generalization_utils.py
│   └── ...
├── scripts/
│   ├── 16_hard_context_ablation.py
│   ├── 18_full_dataset_tfidf.py
│   ├── 19_full_dataset_embeddings.py
│   ├── 20_full_dataset_qwen.py
│   ├── 24_qwen_basic_controls.py
│   ├── 25_unseen_subreddit_generalization.py
│   ├── 28_field_aware_tfidf.py
│   ├── 29_selective_context_routing.py
│   ├── 30_bert_cross_encoder.py
│   └── ...
└── reports/
    ├── final_results.csv
    ├── full_dataset/
    ├── hard_context_ablation/
    ├── qwen_lora_ablation/
    └── qualitative_analysis/
```

---

## Reproduction

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run lightweight baselines:

```bash
python scripts/18_full_dataset_tfidf.py
python scripts/19_full_dataset_embeddings.py
```

Run the BERT cross-encoder:

```bash
python scripts/30_bert_cross_encoder.py --mode all --resume --skip-completed
```

Run the hard-context diagnostic:

```bash
python scripts/16_hard_context_ablation.py
```

GPU experiments require a CUDA-capable machine.

---

## References

- Khodak, M., Saunshi, N., & Vodrahalli, K. (2018). *A Large Self-Annotated Corpus for Sarcasm*. LREC.
- Hazarika, D. et al. (2018). *CASCADE: Contextual Sarcasm Detection in Online Discussion Forums*. COLING.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
- Hu, E. J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR.
