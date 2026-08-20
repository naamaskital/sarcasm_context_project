# Course and Presentation Alignment

This document makes explicit how the final repository preserves the original class presentation while extending it using concepts from **Advanced Models of Language Understanding**.

## 1. Original class-presentation commitments

The presentation framed sarcasm detection through:
- Comment Only
- Context Only
- Context + Comment

and proposed:
- BERT / RoBERTa encoder-only models
- Qwen 1.5B / scaling
- error analysis
- prompt formatting

The final project keeps these commitments and extends them without changing the topic.

| Presentation item | Final implementation |
|---|---|
| Three input conditions | used across TF-IDF, MiniLM, BERT, RoBERTa, DeBERTa, FLAN-T5, and full-data Qwen where applicable |
| Encoder-only | MiniLM frozen + BERT + RoBERTa + DeBERTa cross-encoders |
| Qwen / scaling | Qwen2.5-0.5B + LoRA and Qwen2.5-1.5B controlled scaling |
| Error analysis | helped/hurt categories, curated examples, probability-level behavior analysis |
| Prompt formatting | structured Context/Reply vs plain concatenation |

## 2. Lecturer feedback

The lecturer suggested a modern GPT-style model such as Qwen/Llama, comparison with encoder-only approaches, and multiple model sizes at least in a basic experiment.

The final design covers all three and makes the scaling question stronger through:

`ContextGain = F1(true context) - F1(comment only)`

`ContextSensitivity = F1(true context) - F1(random context)`

## 3. Course concept: representation progression

The project moves through:
1. TF-IDF lexical cues
2. frozen MiniLM contextual sentence representations
3. BERT/RoBERTa/DeBERTa joint token-level encoders
4. FLAN-T5 encoder-decoder architecture
5. Qwen decoder-only LLM

This progression tests whether context utility depends on the model's interaction mechanism rather than merely on model strength.

## 4. Course concept: Transformer architecture families

The syllabus distinguishes:
- Encoder-only / Masked LM
- Encoder-Decoder / Text-to-Text
- Decoder-only / GPT-style

The final project explicitly covers all three:

### Encoder-only
- BERT-base
- RoBERTa-base
- DeBERTa-v3-base

### Encoder-Decoder
- FLAN-T5-base

### Decoder-only
- Qwen2.5-0.5B
- Qwen2.5-1.5B scaling control

Research question:

> Which Transformer family actually benefits from conversational context, and which family is most sensitive to whether that context is correct?

This is a course-driven architectural hypothesis, not a model leaderboard.

## 5. Course concept: model scale

The project uses the entire usable SARC corpus for the main supervised experiments and separately compares two Qwen sizes.

The question is not simply whether 1.5B has higher F1, but whether scale increases ContextGain and ContextSensitivity.

## 6. Course concept: evaluation and good benchmarks

The project uses:
- deterministic IID split
- subreddit-disjoint split
- Macro-F1, sarcastic F1, Accuracy
- paired bootstrap CIs
- true/random/same-subreddit/semantic wrong-context perturbations

The unseen-subreddit protocol directly operationalizes Task learning vs Dataset learning.

## 7. Course concept: PEFT / LoRA

Qwen is adapted using LoRA with rank 8, alpha 16, dropout 0.1, and q_proj/v_proj targets. PEFT is used because it makes the central decoder experiment feasible, not as a detached demonstration.

## 8. Course concept: interpretability

The project focuses on behavioral interpretability because it naturally matches the research question.

For each example, it can measure changes in `P(sarcastic)` under:
- comment only -> true context
- true context -> random wrong context
- true context -> same-subreddit wrong context

Examples are grouped into helped, hurt, irrelevant, and sensitive categories and characterized by length, semantic similarity, `/s`, and subreddit.

## 9. Counterfactual-style analysis

Wrong-context interventions test what the model is actually using:
- random wrong parent
- same-subreddit wrong parent
- semantically similar wrong parent

The semantic hard-negative result motivated stronger cross-encoders and architecture-family comparison.

## 10. Failure-driven method development

### TF-IDF failure
Naive context concatenation hurts.

Hypothesis: lexical dilution.

Action: field-aware TF-IDF with validation-tuned context weighting.

### Context-helped vs context-hurt failure pattern
Context helps some examples and hurts others.

Hypothesis: context is especially useful when reply-only evidence is ambiguous.

Action: selective context routing tuned on validation only.

These methods are derived from prior results rather than added arbitrarily.

## 11. Why DeBERTa and FLAN-T5 were added

### DeBERTa
Adding a stronger encoder-only model tests whether the cross-encoder conclusion generalizes beyond BERT/RoBERTa and whether stronger relational encoding improves context use.

### FLAN-T5
Adding an encoder-decoder model completes the architecture families discussed in the course and turns architecture itself into a research variable.

This is the intended creative extension: course material generates a new hypothesis that is tested inside the same sarcasm/context problem.

## 12. Topics intentionally excluded

RAG, RLHF, agents, and unrelated decoding experiments are not part of the final core because they do not naturally test the project's central question.

## 13. Final intended impression

> class question -> large-scale baseline -> unexpected result -> qualitative/quantitative error analysis -> new representation hypothesis -> multiple justified architecture families -> hard context controls -> scale -> unseen-community benchmark -> behavioral interpretability -> targeted fixes.

That sequence is also the intended structure of the final report.