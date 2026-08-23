# Verified Results

This file collects the reproducible results currently stored in the repository.

## 1. Full-dataset IID results

### TF-IDF + Logistic Regression

| Input | Accuracy | Macro-F1 | Sarcastic F1 |
|---|---:|---:|---:|
| Comment Only | 0.7246 | **0.7242** | 0.7134 |
| Context Only | 0.5857 | 0.5856 | 0.5819 |
| Context + Comment | 0.7049 | 0.7045 | 0.6940 |

**Context gain:** `0.7045 - 0.7242 = -0.0197` Macro-F1.

Naive lexical concatenation hurts performance, motivating field-aware and selective-context follow-up experiments.

### MiniLM + linear classifier

| Input | Accuracy | Macro-F1 | Sarcastic F1 |
|---|---:|---:|---:|
| Comment Only | 0.6673 | 0.6673 | 0.6643 |
| Context Only | 0.5917 | 0.5917 | 0.5896 |
| Dual embeddings | **0.6719** | **0.6719** | **0.6690** |

**Context gain:** approximately `+0.0046` Macro-F1.

### BERT-base cross-encoder

| Input | Accuracy | Macro-F1 | Sarcastic F1 |
|---|---:|---:|---:|
| Comment Only | 0.7783 | 0.7783 | 0.7753 |
| Context Only | 0.6099 | 0.6097 | 0.6182 |
| Context + Comment | **0.7884** | **0.7884** | **0.7882** |

**Context gain:** approximately `+0.0101` Macro-F1.

BERT is the strongest verified full-data result currently stored in the repository.

---

## 2. Qwen2.5-0.5B + LoRA controlled experiment

This experiment uses a smaller balanced protocol and should not be compared numerically with the full-dataset BERT experiment as if they were the same split.

| Input | Accuracy | Macro-F1 | Sarcastic F1 |
|---|---:|---:|---:|
| Context Only | 0.5660 | 0.5613 | 0.6069 |
| Comment Only | 0.6830 | 0.6830 | 0.6801 |
| True Context + Comment | **0.6850** | **0.6850** | **0.6859** |
| Random Context + Comment | 0.6530 | 0.6528 | 0.6615 |

The true-context condition is only slightly above comment-only, but replacing the real context with unrelated context causes a clear drop.

---

## 3. Hard context ablation

The classifier is trained once on true context + comment. At test time, only the context is replaced.

| Test context | Accuracy | Macro-F1 | 95% paired bootstrap delta vs. true |
|---|---:|---:|---:|
| True context | 0.6540 | **0.6540** | – |
| Random wrong context | 0.6375 | 0.6375 | [0.0035, 0.0293] |
| Same-subreddit wrong context | 0.6390 | 0.6390 | [0.0020, 0.0290] |
| Semantically similar wrong context | 0.6540 | 0.6540 | [-0.0113, 0.0118] |

Interpretation: random and same-community replacements hurt, but semantically similar wrong context does not. The frozen embedding classifier therefore appears to exploit **semantic compatibility** more reliably than the exact parent–reply dependency.

---

## 4. Qwen scaling controls

| Model | Comment Macro-F1 | True-context Macro-F1 | Random-context Macro-F1 | Context gain | Context sensitivity |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 0.4350 | 0.4337 | 0.4213 | -0.0014 | 0.0124 |
| Qwen2.5-1.5B-Instruct | 0.4603 | 0.5249 | 0.5179 | +0.0646 | 0.0070 |

The larger model benefits more from the combined input in this control, but the small true-vs-random gap shows that higher context gain does not automatically imply stronger dependence on the exact context.

---

## Overall finding

The experiments support a consistent conclusion:

> **Context is not uniformly useful. Its value depends on representation, architecture, and the individual example.**

The reply itself carries most of the signal, lexical concatenation can hurt, semantic representations can gain modestly, and joint token-level interaction gives the strongest verified full-dataset improvement.