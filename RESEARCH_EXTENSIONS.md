# Advanced Research Extensions

This file fixes the research questions and metrics **before** the remaining GPU results are observed.

## 1. Unseen-subreddit generalization

### Question
Does a sarcasm detector learn the task itself, or does it exploit community-specific Reddit shortcuts?

### Protocol
`src/subreddit_generalization_utils.py` creates a deterministic group-disjoint split using `subreddit` as the grouping variable. No subreddit may appear in more than one of train, validation, or test.

`scripts/25_unseen_subreddit_generalization.py` compares:

- TF-IDF: comment only vs context + comment
- MiniLM: comment only vs separate context/comment embeddings

### Primary quantity

`ContextGain_unseen = MacroF1(context) - MacroF1(comment only)`

The comparison with the ordinary IID split asks whether context becomes more or less useful when community identity cannot be reused across train and test.

### Interpretation
A larger context gain on unseen subreddits would support the hypothesis that conversational information generalizes better than subreddit-specific lexical shortcuts. A smaller or negative gain is also informative: context itself may be community-dependent or noisy.

---

## 2. Does scale improve context utilization?

### Question
Does increasing decoder-model size improve the ability to exploit the *correct conversational relation*, rather than merely improving average accuracy?

### Protocol
`scripts/24_qwen_basic_controls.py` evaluates Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct on the same balanced held-out subset with identical few-shot demonstrations under:

- comment only
- true context + comment
- random context + comment

### Primary quantities

`ContextGain = MacroF1(true context) - MacroF1(comment only)`

`ContextSensitivity = MacroF1(true context) - MacroF1(random context)`

The script writes `qwen_context_utilization_by_scale.csv` so scaling is interpreted in terms of context use, not only raw model performance.

---

## 3. Behavioral interpretability: when does context matter?

### Question
Which examples are affected by context, and what properties characterize them?

### Protocol
The full-data prediction files store `P(sarcastic)` for comment-only, true-context, random-context, and same-subreddit-wrong-context conditions.

`scripts/26_behavioral_context_interpretability.py` computes per example:

- `DeltaP_comment_to_true = P(sarcastic | true context, reply) - P(sarcastic | reply)`
- `DeltaP_true_to_random = P(sarcastic | true context, reply) - P(sarcastic | random context, reply)`
- analogous same-subreddit perturbation

It groups examples into behavioral categories:

- context helped
- context hurt
- context irrelevant
- context sensitive
- other

It then characterizes these groups using:

- comment length
- context length
- context/reply semantic similarity
- explicit sarcasm markers such as `/s`
- subreddit

Thresholds are fixed in code before the final Qwen results: `0.05` for probability-level irrelevance and `0.25` for strong context sensitivity.

---

## Scope discipline

The project deliberately does **not** add unrelated course topics such as RAG, RLHF, agents, or decoding-strategy comparisons. The extensions above all answer the same central question:

> Do models genuinely use conversational context for sarcasm detection, and under what conditions?

Encoder-only cross-encoder experiments and LoRA-rank ablations remain optional second-priority extensions if compute/time allows; they are not required for the core research story.
