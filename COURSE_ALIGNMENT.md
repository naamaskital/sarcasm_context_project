# Course and Presentation Alignment

This document makes explicit how the final repository preserves the original class presentation while extending it using concepts from **Advanced Models of Language Understanding**.

## 1. Original class-presentation commitments

The presentation framed the task as sarcasm detection with conversational context and proposed three controlled inputs:

- Comment Only
- Context Only
- Context + Comment

The final project retains these three conditions across the main model families whenever the architecture permits a direct comparison.

The presentation also listed the following planned extensions:

- encoder-only BERT/RoBERTa
- Qwen 1.5B / model scaling
- manual/error analysis
- prompt formatting

These are represented in the final repository as:

| Presentation item | Final implementation |
|---|---|
| Encoder-only | MiniLM frozen encoder + BERT cross-encoder + RoBERTa cross-encoder |
| Qwen / scaling | Qwen2.5-0.5B + LoRA and Qwen2.5-1.5B controlled scaling experiment |
| Error analysis | reproducible helped/hurt categories + curated examples + behavioral probability analysis |
| Prompt formatting | structured Context/Reply vs plain concatenation in Qwen basic controls |

The final work therefore continues the original project rather than replacing it with a new topic.

---

## 2. Lecturer feedback

The lecturer specifically suggested:

- a modern GPT-style model such as Llama/Qwen;
- ideally a comparison against encoder-only models;
- preferably multiple model sizes, at least for the basic experiment.

The final design directly implements all three:

- Qwen2.5-0.5B-Instruct + LoRA on the full supervised corpus;
- MiniLM, BERT-base, and RoBERTa-base encoder-side comparisons;
- Qwen2.5-0.5B vs 1.5B under an identical few-shot evaluation protocol.

The size comparison is strengthened beyond a raw-accuracy leaderboard by defining:

`ContextGain = F1(true context) - F1(comment only)`

`ContextSensitivity = F1(true context) - F1(random context)`

This turns the lecturer's model-size suggestion into a focused research question about context utilization.

---

## 3. Course concept: lexical and contextual representations

Relevant progression in the project:

1. TF-IDF lexical baseline
2. frozen MiniLM contextual sentence representations
3. BERT/RoBERTa joint token-level contextual representations
4. Qwen decoder-style language model

Research purpose:

> Determine whether the usefulness of context depends on the representation and interaction mechanism rather than simply on adding more text.

This progression was motivated empirically: TF-IDF context concatenation hurt, whereas MiniLM dual embeddings produced a small improvement.

---

## 4. Course concept: Transformer self-attention, Encoder vs Decoder

The project explicitly contrasts:

- **encoder-only / Masked-LM lineage:** BERT, RoBERTa
- **decoder-only / GPT lineage:** Qwen

BERT and RoBERTa receive context and reply as a true text pair, allowing bidirectional self-attention across both fields.

Qwen receives explicitly formatted context/reply sequences and is adapted with LoRA.

The architectural comparison therefore serves the linguistic question:

> Which interaction mechanism is better able to detect a pragmatic relation that may only emerge from the combination of parent message and reply?

---

## 5. Course concept: model scale

The syllabus discusses the effect of model size and amount of data on capabilities.

The project uses the entire usable SARC corpus for its main supervised experiments and separately compares Qwen 0.5B and 1.5B.

The scaling experiment asks a stronger question than whether the larger model has higher F1:

> Does increasing model capacity improve the ability to exploit the *correct conversational context*?

This is measured through ContextGain and ContextSensitivity.

---

## 6. Course concept: evaluation and good benchmarks

The syllabus emphasizes:

- classification evaluation;
- benchmark quality;
- the distinction between task learning and dataset learning.

The project responds with two evaluation protocols:

### IID benchmark
A deterministic stratified 80/10/10 split over 1,010,771 usable examples.

### Subreddit-disjoint benchmark
No subreddit appears in more than one split.

Research question:

> Does the model truly generalize sarcasm/context behavior, or does it exploit community-specific Reddit shortcuts?

The project also uses Macro-F1, sarcastic-class F1, Accuracy, paired bootstrap confidence intervals, and paired context perturbations.

---

## 7. Course concept: PEFT / LoRA

The Qwen supervised model uses LoRA rather than full fine-tuning:

- rank 8
- alpha 16
- dropout 0.1
- q_proj / v_proj targets

This applies PEFT directly to the core project rather than adding a detached PEFT demonstration.

The main goal is not to benchmark every possible LoRA rank, but to make modern decoder adaptation computationally feasible while retaining a clear scientific comparison.

---

## 8. Course concept: interpretability

The syllabus discusses interpretation at behavioral, representation, and weight levels as well as linguistic capabilities and task heuristics.

The project focuses on **behavioral interpretability**, because it directly fits the research question.

For each example, the analysis can measure how `P(sarcastic)` changes under:

- comment only -> true context
- true context -> random wrong context
- true context -> same-subreddit wrong context

Examples are categorized as:

- context helped
- context hurt
- context irrelevant
- context sensitive

Their properties are then examined by:

- comment/context length
- semantic similarity
- explicit `/s` marker
- subreddit

This turns interpretability into the question:

> **When does context actually matter?**

---

## 9. Counterfactual-style analysis and controlled ablations

Although the project does not add unrelated interpretability machinery, it uses behaviorally meaningful counterfactual interventions:

- random wrong parent message
- wrong parent from the same subreddit
- semantically similar but wrong parent

This is stronger than merely comparing `context` vs `no context`, because it asks what information in the context is actually being used.

The semantic hard-negative finding generated a new architecture hypothesis: separate embeddings may capture semantic compatibility without exact conversational dependence. This directly motivated BERT/RoBERTa cross-encoding and Qwen evaluation.

---

## 10. Failure-driven method development

A central creative component is that follow-up methods are derived from observed failures.

### Example A: TF-IDF context hurts
Observation: naive context concatenation reduces full-test Macro-F1.

Hypothesis: lexical dilution/noise from treating context and reply as one field.

Action: `scripts/28_field_aware_tfidf.py` tests separate feature spaces and validation-tuned context weighting.

### Example B: context helps some examples and hurts others
Observation: thousands of helped and hurt cases exist.

Hypothesis: context may be valuable mainly when reply-only evidence is ambiguous.

Action: `scripts/29_selective_context_routing.py` tunes an uncertainty threshold on validation and selectively invokes the context-aware prediction.

These are not arbitrary extra models; they test explanations generated by prior results.

---

## 11. Topics intentionally not added

The course also covers RAG, RLHF, agents, decoding strategies, and additional generative models.

They are intentionally excluded from the final experimental core because they do not naturally test the project's central question about conversational context in sarcasm classification.

The project demonstrates course understanding by **selecting the relevant concepts and using them to deepen one coherent scientific question**, rather than by attaching every syllabus topic to the repository.

---

## 12. Final intended impression

The repository should show this progression:

> class-presentation question -> large-scale controlled baseline -> unexpected results -> qualitative and quantitative error analysis -> hypotheses about representation and shortcuts -> stronger encoder/decoder models -> hard counterfactual controls -> scaling as context utilization -> unseen-community benchmark -> behavioral interpretability -> targeted fixes derived from failure analysis.

This is the organizing principle for the final report as well.
