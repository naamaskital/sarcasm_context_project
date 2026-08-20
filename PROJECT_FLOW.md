# Project Flow: Hypothesis -> Experiment -> Analysis -> New Hypothesis

This document records the intended scientific narrative of the project.

## Starting point: the class presentation

The class presentation introduced:

> Does conversational context improve sarcasm detection in Reddit replies?

with:
- Comment Only
- Context Only
- Context + Comment

It also proposed encoder-only models, scaling, prompt formatting, and error analysis. The final project preserves that starting point and extends it through evidence-driven research loops.

---

## Loop 1: Is context useful at all?

### H1
The reply contains much of the sarcasm signal, but context may add pragmatic information.

### Experiments
TF-IDF + Logistic Regression and frozen MiniLM, each under the three original input conditions.

### Result
TF-IDF is harmed by naive context concatenation, while MiniLM dual embeddings improve slightly.

### New hypothesis
The effect of context depends on representation and integration.

---

## Loop 2: Can failure analysis suggest a better representation?

### Failure
Long parent messages can dilute short diagnostic replies in a bag-of-words representation.

### H2
Context may still be useful if its features are kept separate from the reply.

### Action
`scripts/28_field_aware_tfidf.py`

Separate context/reply lexical feature spaces with validation-tuned context weighting.

### Additional behavior-derived action
`scripts/29_selective_context_routing.py`

If context helps only on ambiguous replies, route to the context-aware predictor only when reply-only confidence is low. Threshold selection is validation-only.

---

## Loop 3: Does separate semantic representation capture the true relation?

### Observation
MiniLM dual embeddings improve slightly.

### Stronger test
Replace true context with random, same-subreddit, and semantically similar wrong context.

### Result
True context beats random and same-subreddit wrong context, but not semantically similar wrong context.

### H3
Separate embeddings may capture semantic compatibility without exact conversational dependence.

### Action
Move to joint token-level interaction.

---

## Loop 4: How much does joint encoder interaction matter?

### H4
A true cross-encoder should better model the relation between parent message and reply than separate sentence embeddings.

### Encoder progression
- BERT-base
- RoBERTa-base
- DeBERTa-v3-base

All use Comment Only / Context Only / Context + Comment.

### Why several encoders?
They are not redundant leaderboard entries. They test whether the conclusion is stable across increasingly strong encoder-only architectures rather than being an artifact of one model.

### Analysis
Compare overall Macro-F1, ContextGain, and hard/context-dependent examples.

---

## Loop 5: Does Transformer family change context utilization?

### Course-driven hypothesis
The course distinguishes Encoder-only, Encoder-Decoder, and Decoder-only Transformers. A conversational relation may be handled differently by each family.

### H5
Architecture family affects how much useful signal is extracted from context and how sensitive the model is to incorrect context.

### Models
- Encoder-only: BERT / RoBERTa / DeBERTa
- Encoder-Decoder: FLAN-T5-base
- Decoder-only: Qwen2.5

### Key quantities
- Comment-only Macro-F1
- True-context Macro-F1
- ContextGain
- ContextSensitivity where wrong-context evaluation is available

### Research question
Which architecture family benefits from context, rather than merely achieving the highest task score?

---

## Loop 6: Does a modern decoder use context differently?

### H6
A decoder model adapted with LoRA may learn a different interaction pattern from encoder-only and encoder-decoder models.

### Experiment
Qwen2.5-0.5B + LoRA on the full corpus with all three original input conditions.

Then evaluate the context-trained adapter under true, random, and same-subreddit wrong context.

### Analysis
Macro-F1, bootstrap CIs, changed-prediction counts, and per-example probability shifts.

---

## Loop 7: Does model scale improve context utilization?

### H7
A larger decoder may be better not just at classification, but at exploiting the correct conversational relation.

### Experiment
Qwen2.5-0.5B vs Qwen2.5-1.5B under identical few-shot conditions.

### Metrics
`ContextGain = MacroF1(true) - MacroF1(comment only)`

`ContextSensitivity = MacroF1(true) - MacroF1(random)`

---

## Loop 8: Did the model learn sarcasm, or Reddit communities?

### H8
IID splits may preserve subreddit-specific shortcuts.

### Experiment
Subreddit-disjoint train/validation/test split.

### Question
Does context become more useful, less useful, or equally useful on unseen communities?

---

## Loop 9: When should context be used?

### Observation
There are many context-helped and context-hurt examples.

### H9
Context is especially useful for ambiguous replies and can be harmful when it introduces irrelevant signal.

### Behavioral interpretability
Analyze probability shifts, text lengths, semantic similarity, `/s`, and subreddit.

### Targeted solution
Selective context routing tuned on validation only.

---

## Final scientific narrative

> Initial hypothesis -> controlled baseline -> unexpected result -> example/error analysis -> representation hypothesis -> stronger model family -> counterfactual evaluation -> scaling/generalization -> behavioral interpretability -> targeted corrective action.

The final report should preserve this causal sequence even when hypotheses are rejected. Negative results remain scientifically useful because they motivate the next controlled test.