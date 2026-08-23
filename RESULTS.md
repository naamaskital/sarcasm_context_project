# Final Report Results

This file mirrors the quantitative results used in the final scientific report. The compact machine-readable version is `reports/final_report_results.csv`.

## 1. Full-dataset IID comparison

Macro-F1 on the fixed test split of 101,078 examples:

| Model | Comment Only | Context Only | Context + Comment |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | **0.7242** | 0.5856 | 0.7045 |
| MiniLM + linear classifier | 0.6673 | 0.5917 | **0.6719** |
| BERT-base cross-encoder | 0.7783 | 0.6097 | **0.7884** |
| RoBERTa-base cross-encoder | 0.7840 | 0.6159 | **0.7955** |
| DeBERTa-v3-base cross-encoder | 0.7901 | 0.6205 | **0.8030** |
| FLAN-T5-base | 0.7561 | 0.5981 | **0.7705** |
| Qwen2.5-0.5B + LoRA | 0.7385 | 0.6010 | **0.7521** |

The strongest result in the final report is DeBERTa-v3-base with context + comment, Macro-F1 **0.8030**.

## 2. Failure-driven lexical follow-ups

| Method | Macro-F1 |
|---|---:|
| Comment-only TF-IDF baseline | 0.7242 |
| Field-aware TF-IDF | 0.7297 |
| Selective context routing | **0.7317** |

These experiments test whether context is more useful when its feature space is kept separate or used only on uncertain examples.

## 3. Hard context ablation

The MiniLM context-aware classifier is trained once on true context + comment. At test time, only the context is replaced.

| Test context | Macro-F1 | 95% paired bootstrap delta vs. true |
|---|---:|---:|
| True context | **0.6540** | – |
| Random wrong context | 0.6375 | [0.0035, 0.0293] |
| Same-subreddit wrong context | 0.6390 | [0.0020, 0.0290] |
| Semantically similar wrong context | 0.6540 | [-0.0113, 0.0118] |

Semantically similar wrong contexts are selected using normalized `all-MiniLM-L6-v2` embeddings and cosine similarity, with self-matches and identical context strings excluded. Bootstrap confidence intervals use 1,000 paired resamples.

Interpretation: the frozen embedding classifier is sensitive to random and same-community context replacement, but a semantically similar wrong context can behave like the true parent. This supports semantic compatibility more strongly than exact conversational dependence.

## 4. Qwen2.5-0.5B + LoRA controlled context ablation

This is a separate balanced protocol from the full-dataset Qwen run: **3,000 train / 500 validation / 1,000 test**, 5 epochs.

| Input | Accuracy | Macro-F1 | Sarcastic F1 |
|---|---:|---:|---:|
| Context Only | 0.5660 | 0.5613 | 0.6069 |
| Comment Only | 0.6830 | 0.6830 | 0.6801 |
| True Context + Comment | **0.6850** | **0.6850** | **0.6859** |
| Random Context + Comment | 0.6530 | 0.6528 | 0.6615 |

The true-context condition is only slightly above comment-only, but unrelated random context causes a clear drop.

## 5. Qwen scaling / context-utilization control

A balanced 1,000-example held-out sample is evaluated in a fixed 4-shot protocol (two demonstrations per class).

| Model | Comment Macro-F1 | True-context Macro-F1 | Random-context Macro-F1 | Context gain | Context sensitivity |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 0.4350 | 0.4337 | 0.4213 | -0.0014 | 0.0124 |
| Qwen2.5-1.5B-Instruct | 0.4603 | 0.5249 | 0.5179 | +0.0646 | 0.0070 |

The larger model gains more from combined input, while the true-vs-random gap remains small. Higher context gain therefore does not necessarily imply stronger dependence on the exact conversational parent.

## 6. Unseen-subreddit generalization

Macro-F1 on subreddit-disjoint evaluation:

| Model | Comment Only | Context-aware |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.7099 | 0.6887 |
| MiniLM + linear classifier | 0.6578 | 0.6598 |

The context contribution is weaker under community shift, reinforcing that context is not uniformly useful across distributions.

## Overall finding

> **Context is not uniformly useful. Its value depends on representation, architecture, protocol, and the individual example.**

The reply carries most of the signal, naive lexical concatenation can hurt, semantically compatible wrong context can mimic the true parent for frozen embeddings, and stronger joint models benefit more consistently from context.
