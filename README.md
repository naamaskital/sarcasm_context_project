# Sarcasm Detection in the Context Test

Final project for **Advanced Models of Language Understanding**.

## Core research question

The project began with the class-presentation question:

> **Does conversational context improve sarcasm detection in Reddit replies?**

The original controlled inputs are preserved throughout the final project:

1. `Comment Only`
2. `Context Only`
3. `Context + Comment`

The final version develops that starting point into a hypothesis-driven process:

> **Do language models genuinely exploit the conversational relation between a Reddit parent message and its reply, or do they mainly exploit lexical, semantic, architectural, model-size, or community-specific shortcuts?**

The repository is organized as:

> **hypothesis -> experiment -> result/error analysis -> new hypothesis -> targeted follow-up**

rather than as a model leaderboard.

See:
- `PROJECT_FLOW.md` - full research narrative
- `COURSE_ALIGNMENT.md` - mapping to the class presentation, lecturer feedback, and syllabus
- `RESEARCH_EXTENSIONS.md` - pre-specified advanced hypotheses
- `GPU_RUN.md` - exact execution order

---

## Dataset and evaluation protocols

Main source: original SARC Reddit corpus (`train-balanced-sarcasm.csv`).

### IID full-data protocol
- raw rows: **1,010,826**
- usable rows: **1,010,771**
- train: **808,616**
- validation: **101,077**
- test: **101,078**
- seed: **42**

### Unseen-subreddit protocol
`src/subreddit_generalization_utils.py` creates a group-disjoint split in which train, validation, and test contain different subreddits. This tests **task learning vs dataset/community learning**.

---

# Research flow

## Stage 1 - Is the reply itself enough?

### Hypothesis
Many sarcastic replies contain strong local lexical signals, so context may not always be necessary.

### Models
- TF-IDF + Logistic Regression
- all-MiniLM-L6-v2 frozen embeddings + linear classifier

### Completed full-data results

#### TF-IDF
| Input | Accuracy | Macro-F1 | Sarcastic F1 |
|---|---:|---:|---:|
| comment only | **0.7246** | **0.7242** | **0.7134** |
| context only | 0.5857 | 0.5856 | 0.5819 |
| context + comment | 0.7049 | 0.7045 | 0.6940 |

Naive context concatenation hurts by about **0.0197 Macro-F1**.

#### MiniLM
| Input | Accuracy | Macro-F1 | Sarcastic F1 |
|---|---:|---:|---:|
| comment only | 0.6675 | 0.6675 | 0.6645 |
| context only | 0.5916 | 0.5916 | 0.5890 |
| dual embeddings | **0.6721** | **0.6720** | **0.6689** |

Separate semantic representations help slightly.

### New hypothesis
Context usefulness depends on **representation and interaction**, not only on adding more text.

---

## Stage 2 - Failure-driven fixes

### TF-IDF dilution hypothesis
A long parent message may dilute a short diagnostic reply.

Targeted solution:
`scripts/28_field_aware_tfidf.py`

It uses separate lexical feature spaces and validation-tuned context weighting.

### Selective-context hypothesis
Context may help mainly when reply-only evidence is ambiguous.

Targeted solution:
`scripts/29_selective_context_routing.py`

It chooses the context-aware prediction only when reply-only confidence is low; thresholds are selected on validation only.

---

## Stage 3 - From frozen representations to joint interaction

### Observation
MiniLM dual embeddings help slightly, but the hard-negative experiment shows a critical limitation: a semantically similar **wrong** context can perform almost as well as the true parent message.

### Hypothesis
Separate embeddings may capture topic/semantic compatibility without modeling the exact conversational relation.

### Encoder-only progression
- **MiniLM** - frozen separate sentence representations
- **BERT-base** - canonical fine-tuned encoder-only cross-encoder
- **RoBERTa-base** - stronger BERT-style encoder under the same protocol
- **DeBERTa-v3-base** - more modern encoder-only architecture used to test whether stronger relational encoding improves context utilization

Scripts:
```text
scripts/19_full_dataset_embeddings.py
scripts/30_bert_cross_encoder.py
scripts/27_roberta_cross_encoder.py
scripts/31_deberta_cross_encoder.py
```

BERT, RoBERTa, and DeBERTa all use the three original presentation conditions and encode `context + comment` jointly.

Research question:

> **Does increasingly expressive joint bidirectional interaction improve the ability to use the correct conversational relation?**

---

## Stage 4 - Encoder-only vs Encoder-Decoder vs Decoder-only

The course explicitly distinguishes the three main Transformer families. The project therefore adds a controlled architectural comparison rather than only adding stronger models of the same type.

### Encoder-only
BERT / RoBERTa / DeBERTa

### Encoder-Decoder
**FLAN-T5-base** via `scripts/32_flan_t5_encoder_decoder.py`

FLAN-T5 receives the same three input conditions and is evaluated as a binary classifier. It tests whether an encoder-decoder architecture uses context differently from encoder-only and decoder-only models.

### Decoder-only
Qwen2.5

Research question:

> **Which Transformer family benefits most from conversational context, and which family is most sensitive to the correctness of that context?**

This turns architecture into a research variable rather than a leaderboard category.

---

## Stage 5 - Does the model need the *true* context?

Counterfactual test conditions:
1. true context
2. random wrong context
3. same-subreddit wrong context
4. semantically similar wrong context

Completed MiniLM hard-context diagnostic:

| Context condition | Macro-F1 | paired 95% CI for true - alternative |
|---|---:|---:|
| true | 0.6540 | - |
| random wrong | 0.6375 | [0.0035, 0.0293] |
| same-subreddit wrong | 0.6390 | [0.0020, 0.0290] |
| semantic-similar wrong | 0.6540 | [-0.0113, 0.0118] |

This motivates stronger joint architectures and Qwen.

---

## Stage 6 - Modern decoder + PEFT

**Qwen2.5-0.5B-Instruct + LoRA**

Full-data training conditions:
- comment only
- context only
- context + comment

LoRA:
- rank 8
- alpha 16
- dropout 0.1
- q_proj / v_proj

Post-training ablation evaluates the same context-trained adapter on true, random, and same-subreddit wrong context and reports paired bootstrap intervals, prediction changes, and `P(sarcastic)` shifts.

---

## Stage 7 - Does scale improve context utilization?

`scripts/24_qwen_basic_controls.py` compares Qwen2.5-0.5B and Qwen2.5-1.5B under identical few-shot conditions.

Primary measures:

`ContextGain = MacroF1(true context) - MacroF1(comment only)`

`ContextSensitivity = MacroF1(true context) - MacroF1(random context)`

Research question:

> **Does scale improve the ability to exploit conversational relationships, not merely raw task accuracy?**

The same experiment also tests structured Context/Reply formatting versus plain concatenation.

---

## Stage 8 - Task learning or dataset learning?

The subreddit-disjoint benchmark asks:

> **Did the model learn sarcasm/context behavior, or community-specific Reddit shortcuts?**

`scripts/25_unseen_subreddit_generalization.py` compares comment-only and context-aware behavior on subreddits unseen during training.

---

## Stage 9 - Behavioral interpretability

`scripts/26_behavioral_context_interpretability.py` measures per-example probability shifts:
- comment -> true context
- true -> random context
- true -> same-subreddit wrong context

Examples are categorized as:
- context helped
- context hurt
- context irrelevant
- context sensitive

Groups are analyzed by:
- reply length
- context length
- semantic similarity
- `/s` marker
- subreddit

Research question:

> **When does context actually matter?**

---

## Stage 10 - Qualitative evidence and new hypotheses

`scripts/23_qualitative_error_analysis.py` identifies reproducible success/failure categories and deterministic example samples.

Current counts include:
- TF-IDF context helped: **9,652**
- TF-IDF context hurt: **11,646**
- MiniLM context helped: **4,361**
- MiniLM context hurt: **3,899**
- TF-IDF hurt while MiniLM helped on the same example: **417**

Qualitative examples are used to generate follow-up hypotheses, not merely illustrate final numbers.

---

# Model matrix

| Model / method | Family | Adaptation | Research role |
|---|---|---|---|
| TF-IDF + LR | lexical | supervised LR | simple lexical baseline |
| Field-aware TF-IDF | lexical fields | validation-tuned weighting | failure-derived fix |
| MiniLM | encoder representation | frozen + linear head | semantic representation without joint attention |
| BERT-base | encoder-only | full fine-tuning | canonical Masked-LM cross-encoder |
| RoBERTa-base | encoder-only | full fine-tuning | stronger BERT-style control |
| DeBERTa-v3-base | encoder-only | full fine-tuning | tests stronger relational encoder |
| FLAN-T5-base | encoder-decoder | fine-tuned classification | completes Transformer-family comparison |
| Qwen2.5-0.5B | decoder-only | LoRA | modern GPT-style supervised model |
| Qwen2.5-1.5B | decoder-only | controlled few-shot scaling | tests scale as context utilization |
| Selective routing | behavior-driven policy | validation-tuned | tests whether analysis can improve decision strategy |

---

# Course concepts used directly

- lexical vs contextual representations
- self-attention and token interaction
- encoder-only / Masked LM: BERT, RoBERTa, DeBERTa
- encoder-decoder / text-to-text lineage: FLAN-T5
- decoder-only / GPT-style: Qwen
- effect of model scale
- few-shot prompting
- classification evaluation and benchmark design
- Task learning vs Dataset learning
- PEFT / LoRA
- behavioral interpretability
- counterfactual-style perturbations

RAG, RLHF, agents, and unrelated decoding experiments are intentionally excluded because they do not naturally answer the sarcasm/context question.

---

# Execution

CPU / lightweight stages:
```bash
python src/full_dataset_utils.py
python src/subreddit_generalization_utils.py
python scripts/18_full_dataset_tfidf.py
python scripts/19_full_dataset_embeddings.py
python scripts/25_unseen_subreddit_generalization.py
python scripts/28_field_aware_tfidf.py
python scripts/29_selective_context_routing.py
python scripts/23_qualitative_error_analysis.py
```

GPU stages:
```bash
python scripts/24_qwen_basic_controls.py
python scripts/30_bert_cross_encoder.py --mode all --resume --skip-completed
python scripts/27_roberta_cross_encoder.py --mode all --resume --skip-completed
python scripts/31_deberta_cross_encoder.py --mode all --resume --skip-completed
python scripts/32_flan_t5_encoder_decoder.py --mode all --resume --skip-completed
python scripts/20_full_dataset_qwen.py --mode all --resume --skip-completed
python scripts/21_full_dataset_qwen_context_ablation.py
python scripts/26_behavioral_context_interpretability.py
python scripts/23_qualitative_error_analysis.py
python scripts/22_collect_final_results.py
```

See `GPU_RUN.md` for the detailed order and checkpoint behavior.

---

# Repository map

```text
README.md                                        project overview
PROJECT_FLOW.md                                 hypothesis-driven narrative
COURSE_ALIGNMENT.md                             presentation + syllabus alignment
RESEARCH_EXTENSIONS.md                          pre-specified advanced hypotheses
GPU_RUN.md                                      execution guide

src/full_dataset_utils.py                       IID full-corpus split
src/subreddit_generalization_utils.py           subreddit-disjoint split

scripts/16_hard_context_ablation.py             semantic hard-negative diagnostic
scripts/18_full_dataset_tfidf.py                lexical baseline
scripts/19_full_dataset_embeddings.py           MiniLM frozen encoder
scripts/20_full_dataset_qwen.py                 Qwen + LoRA full-data experiment
scripts/21_full_dataset_qwen_context_ablation.py Qwen context perturbation
scripts/23_qualitative_error_analysis.py         qualitative success/failure analysis
scripts/24_qwen_basic_controls.py               scaling + prompt-format controls
scripts/25_unseen_subreddit_generalization.py    generalization benchmark
scripts/26_behavioral_context_interpretability.py behavioral interpretability
scripts/27_roberta_cross_encoder.py              RoBERTa cross-encoder
scripts/28_field_aware_tfidf.py                  failure-driven TF-IDF fix
scripts/29_selective_context_routing.py           behavior-driven routing fix
scripts/30_bert_cross_encoder.py                 BERT cross-encoder
scripts/31_deberta_cross_encoder.py              DeBERTa cross-encoder
scripts/32_flan_t5_encoder_decoder.py            FLAN-T5 encoder-decoder experiment
scripts/22_collect_final_results.py              consolidated final tables
```

## Current conclusions before remaining GPU runs

1. The reply is the strongest individual signal.
2. More text is not automatically better.
3. Representation design changes whether context helps.
4. Semantic compatibility is not the same as exact conversational dependence.
5. These findings motivate stronger joint encoders, a distinct encoder-decoder family, modern decoder models, scale analysis, generalization tests, and behavioral interpretability.

The remaining experiments are intended to confirm, refine, or reject these hypotheses.

## References
- Khodak, M., Saunshi, N., & Vodrahalli, K. (2018). *A Large Self-Annotated Corpus for Sarcasm*. LREC.
- Hazarika, D. et al. (2018). *CASCADE: Contextual Sarcasm Detection in Online Discussion Forums*. COLING.
- Devlin, J. et al. (2019). *BERT*. NAACL.
- Liu, Y. et al. (2019). *RoBERTa*.
- He, P. et al. (2021). *DeBERTa*.
- Chung, H. W. et al. (2022). *Scaling Instruction-Finetuned Language Models*.
- Hu, E. J. et al. (2022). *LoRA*. ICLR.
