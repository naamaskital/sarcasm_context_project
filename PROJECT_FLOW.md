# Project Flow: Hypothesis -> Experiment -> Analysis -> New Hypothesis

This document records the intended scientific narrative of the project.

## Starting point: the class presentation

The class presentation introduced a simple controlled question:

> Does conversational context improve sarcasm detection in Reddit replies?

The initial input conditions were:

- Comment Only
- Context Only
- Context + Comment

The planned extensions presented in class included encoder-only models, model scaling, prompt formatting, and error analysis.

The final project preserves that starting point but turns it into a sequence of research loops.

---

## Loop 1: Is context useful at all?

### Hypothesis H1
The target reply contains much of the sarcasm signal, but the parent context may add useful pragmatic information.

### Experiments
- TF-IDF + Logistic Regression
- MiniLM frozen embeddings + linear classifier
- three original input conditions

### Result
- TF-IDF: context concatenation hurts relative to comment-only.
- MiniLM: separate context/reply embeddings produce a small gain.

### Analysis
The effect of context depends on representation and integration, not only on whether more text is available.

### New hypothesis H2
Naive lexical concatenation introduces noise, while preserving the two fields separately may protect the reply signal.

### Follow-up actions
- field-aware TF-IDF
- joint encoder models

---

## Loop 2: Can a failure-derived representation fix help?

### Observed failure
Short, highly diagnostic replies are sometimes diluted by long parent messages in TF-IDF.

### Hypothesis H2
Context is not necessarily useless; the failure may come from treating parent and reply words as one undifferentiated bag.

### Targeted solution
`scripts/28_field_aware_tfidf.py`

Separate lexical feature spaces are built for comment and context. Context contribution is tuned on validation only.

### Test
Compare:
- comment-only TF-IDF
- naive concatenation
- field-aware combination

### Interpretation
If field-aware TF-IDF recovers the lost performance, the original failure is evidence about representation design rather than evidence that context has no value.

---

## Loop 3: Does separate semantic representation capture the true relation?

### Observation
MiniLM dual embeddings improve slightly, suggesting useful semantic information in context.

### Stronger test
Replace the true parent message at test time with:
- random context
- same-subreddit wrong context
- semantically similar wrong context

### Result
The embedding model distinguishes true context from random and same-subreddit wrong context, but not from a semantically similar wrong context.

### New hypothesis H3
Separate sentence embeddings may exploit semantic compatibility without modeling the exact conversational dependency.

### Follow-up action
Use joint token-level interaction:
- BERT cross-encoder
- RoBERTa cross-encoder
- Qwen decoder model

---

## Loop 4: Does joint token interaction solve the embedding limitation?

### Hypothesis H3
If exact conversational relations require interactions between individual words/tokens across the two messages, a cross-encoder should outperform separate embeddings particularly on context-dependent and hard-negative examples.

### Experiments
- BERT-base cross-encoder
- RoBERTa-base cross-encoder
- comment only / context only / context + comment

### Analysis plan
Compare both overall metrics and qualitative cases where MiniLM failed under semantically similar wrong context.

### Possible conclusions
- If cross-encoders improve: joint interaction appears important.
- If they do not: the failure is deeper than the separate-embedding architecture.

---

## Loop 5: Does a modern decoder use context differently?

### Motivation
Lecturer feedback explicitly suggested a modern GPT-style model such as Qwen/Llama and comparison with encoder-only approaches.

### Hypothesis H4
A modern decoder adapted with LoRA may learn a different form of context/reply interaction than encoder-only models.

### Experiment
Qwen2.5-0.5B-Instruct + LoRA on the full training split under:
- comment only
- context only
- context + comment

Then evaluate the same context-trained adapter under:
- true context
- random context
- same-subreddit wrong context

### Analysis
Use:
- Macro-F1
- paired bootstrap confidence intervals
- changed-prediction counts
- per-example probability shifts

---

## Loop 6: Does model scale improve context utilization?

### Hypothesis H5
A larger decoder may not only improve average task performance; it may be better at using the correct conversational relation.

### Experiment
Qwen2.5-0.5B vs Qwen2.5-1.5B under identical few-shot conditions:
- comment only
- true context + comment
- random context + comment

### Derived metrics

`ContextGain = MacroF1(true context) - MacroF1(comment only)`

`ContextSensitivity = MacroF1(true context) - MacroF1(random context)`

### Research question
Does scale improve context utilization rather than only raw accuracy?

---

## Loop 7: Did the model learn sarcasm, or Reddit communities?

### Course motivation
The evaluation unit distinguishes learning the task from learning the dataset and discusses properties of a good benchmark.

### Hypothesis H6
Random IID splits may allow community-specific lexical shortcuts to transfer from train to test.

### Experiment
Create a subreddit-disjoint benchmark in which no subreddit appears in more than one split.

Compare comment-only and context-aware variants again.

### Derived metric

`ContextGain_unseen = MacroF1(context-aware) - MacroF1(comment-only)`

### Interpretation
A change in context gain between IID and unseen-subreddit evaluation reveals how much the usefulness of context depends on the evaluation distribution.

---

## Loop 8: When should context be used?

### Observation
There are many context-helped examples and many context-hurt examples.

### Hypothesis H7
Context may be most useful when the reply alone is ambiguous or low-confidence.

### Behavioral analysis
Measure per example:
- probability shift from comment-only to true context
- probability shift from true to wrong context
- comment/context length
- semantic similarity
- `/s` markers
- subreddit

Categorize examples as:
- context helped
- context hurt
- context irrelevant
- context sensitive

### Targeted solution
`scripts/29_selective_context_routing.py`

Use the context-aware classifier only when the comment-only model is uncertain. Tune the uncertainty threshold on validation only, then evaluate once on test.

### Research question
Can model behavior analysis lead to a better decision strategy rather than merely explain errors after the fact?

---

## Final scientific narrative

The project is not organized around a leaderboard. Its structure is:

> Initial hypothesis -> controlled baseline -> unexpected result -> error/example analysis -> new hypothesis -> targeted architecture or intervention -> stronger evaluation -> interpretability -> solution derived from observed behavior.

The desired final report should preserve this causal narrative even if some hypotheses are rejected. Negative findings are scientifically useful when they motivate the next controlled test.
