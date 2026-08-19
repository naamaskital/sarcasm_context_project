# Sarcasm Detection with Conversational Context

Final project for **Advanced Models of Language Understanding**.

## Research question

**Do sarcasm models actually use conversational context, or do they mainly exploit extra lexical and semantic cues?**

The project is designed as a sequence of controlled tests rather than a model leaderboard. It asks not only whether context changes performance, but **what kind of contextual information a model is using**.

## Main dataset protocol

The main experiment now uses the original `train-balanced-sarcasm.csv` corpus from `marcbishara/sarcasm-on-reddit`.

- raw rows: **1,010,826**
- usable after removing 55 missing/empty text rows: **1,010,771**
- fixed stratified train split: **808,616**
- validation: **101,077**
- test: **101,078**
- seed: **42**

`src/full_dataset_utils.py` creates this deterministic 80/10/10 split once and caches it locally. All full-dataset model comparisons use the **same split**.

The repository still contains a balanced 2,000-example sample for quick sanity checks and earlier pilot experiments. Those pilot results are kept for the experimental record but are not the main large-scale evaluation.

## Controlled input conditions

The project uses progressively harder controls:

1. **comment only** - target reply only.
2. **context only** - previous Reddit message only.
3. **true context + comment** - the real conversational pair.
4. **random context + comment** - unrelated context.
5. **same-subreddit wrong context** - wrong context from the same community/topic environment.
6. **semantically similar wrong context** - hard negative that looks semantically compatible but is not the true parent message.

The last three conditions are designed to distinguish true conversational dependence from simply receiving more text or topically similar text.

## Methods

### TF-IDF + Logistic Regression

A lexical baseline using unigrams/bigrams and balanced Logistic Regression. It tests whether recurring lexical sarcasm cues are sufficient.

### all-MiniLM-L6-v2 + linear classifier

A frozen Sentence Transformer produces normalized embeddings. `comment_only`, `context_only`, and **separate context/comment embeddings** are compared. Full-corpus embeddings are generated in chunks and stored with memmap so the experiment is practical at one-million-example scale.

### Qwen2.5-0.5B-Instruct + LoRA

The final interaction-aware model is a binary sequence classifier adapted with LoRA:

- `r=8`
- `alpha=16`
- `dropout=0.1`
- target modules: `q_proj`, `v_proj`
- learning rate: `2e-4`
- full-data training: one complete epoch over **808,616** examples per input condition

The full-data script requires CUDA, saves checkpoints every 1,000 optimizer steps, supports resume, saves the final adapter, and evaluates on the complete validation/test sets.

## Completed full-dataset results

### TF-IDF

| Input | Test Accuracy | Test Macro-F1 | Sarcastic F1 |
|---|---:|---:|---:|
| comment only | **0.7246** | **0.7242** | **0.7134** |
| context only | 0.5857 | 0.5856 | 0.5819 |
| context + comment | 0.7049 | 0.7045 | 0.6940 |

A naive context concatenation **hurts** TF-IDF relative to comment-only by about 0.0197 Macro-F1.

### MiniLM

| Input | Test Accuracy | Test Macro-F1 | Sarcastic F1 |
|---|---:|---:|---:|
| comment only | 0.6675 | 0.6675 | 0.6645 |
| context only | 0.5916 | 0.5916 | 0.5890 |
| dual embeddings | **0.6721** | **0.6720** | **0.6689** |

Unlike TF-IDF, keeping context and reply as separate semantic representations produces a small improvement over comment-only. This makes representation design part of the research finding: **context is not automatically useful; how it is integrated matters.**

## Hard-context diagnostic ablation

The controlled embedding experiment trains once on true context + comment and changes only the test context.

| Test context | Accuracy | Macro-F1 | paired 95% bootstrap delta: true - alternative |
|---|---:|---:|---:|
| true context | 0.6540 | 0.6540 | - |
| random wrong | 0.6375 | 0.6375 | [0.0035, 0.0293] |
| same-subreddit wrong | 0.6390 | 0.6390 | [0.0020, 0.0290] |
| semantically similar wrong | 0.6540 | 0.6540 | [-0.0113, 0.0118] |

Same-subreddit hard-negative coverage was **84.7%** and mean semantic hard-negative cosine similarity was **0.4451**.

The nuanced finding is important: the embedding classifier distinguishes true context from random and same-community context, but **not from a semantically similar incorrect context**. It therefore appears to exploit semantic compatibility without reliably modeling the exact conversational dependency.

## Qualitative error analysis

`scripts/23_qualitative_error_analysis.py` turns the saved test predictions into reproducible qualitative categories. The examples are not manually cherry-picked: examples are first selected by a fixed outcome criterion and then deterministically sampled with a fixed seed. Deleted/empty and extremely short/long texts are filtered only for readability.

The analysis includes:

- **TF-IDF context helped**: comment-only is wrong while context+comment is correct.
- **TF-IDF context hurt**: comment-only is correct while context+comment is wrong.
- **MiniLM context helped / hurt** using comment-only vs separate dual embeddings.
- **Representation-matters cases** where naive TF-IDF context hurts on the same example where MiniLM's separate context representation helps.
- **Hard-negative failures**, including cases where true context is correct but a random, same-subreddit, or semantically similar wrong context changes the result.
- **Semantically similar wrong context with unchanged prediction**, which helps illustrate the limitation revealed by the hard-negative aggregate metrics.
- After the GPU run, the same script automatically adds **Qwen context helped / hurt** and Qwen hard-context cases when those prediction files exist.

Run it with:

```bash
python scripts/23_qualitative_error_analysis.py
```

Outputs:

```text
reports/qualitative_analysis/qualitative_category_counts.csv
reports/qualitative_analysis/selected_qualitative_examples.csv
reports/qualitative_analysis/qualitative_analysis_summary.txt
```

The category-count file quantifies how often each qualitative pattern occurs; the selected-examples file is intended for a small report table and targeted manual interpretation.

## Qwen full-dataset status

The full-dataset Qwen code is complete; only GPU execution remains.

Expected final comparison:

| Input | Train | Validation | Test | Test Macro-F1 |
|---|---:|---:|---:|---:|
| comment only | 808,616 | 101,077 | 101,078 | pending GPU |
| true context + comment | 808,616 | 101,077 | 101,078 | pending GPU |

After the context-trained adapter is produced, `scripts/21_full_dataset_qwen_context_ablation.py` evaluates the **same trained model** on true, random, and same-subreddit test contexts and reports paired bootstrap confidence intervals plus context-sensitivity counts. No additional Qwen training is needed for the ablation.

The semantic-nearest hard-negative condition remains the focused diagnostic experiment in `scripts/16_hard_context_ablation.py`; an exact all-pairs search is intentionally not expanded to all 101k test contexts.

## Reproduction

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create the fixed split:

```bash
python src/full_dataset_utils.py
```

Run completed CPU experiments:

```bash
python scripts/18_full_dataset_tfidf.py
python scripts/19_full_dataset_embeddings.py
python scripts/23_qualitative_error_analysis.py
```

On a CUDA machine, run the final Qwen experiments:

```bash
python scripts/20_full_dataset_qwen.py --mode comment_only --resume
python scripts/20_full_dataset_qwen.py --mode context_plus_comment --resume
python scripts/21_full_dataset_qwen_context_ablation.py
python scripts/22_collect_final_results.py
python scripts/23_qualitative_error_analysis.py
```

See **`GPU_RUN.md`** for the exact end-to-end GPU workflow, checkpoint/resume behavior, and expected artifacts.

## Key files

```text
src/full_dataset_utils.py                        fixed 1.01M-example split
scripts/18_full_dataset_tfidf.py                full-data lexical baseline
scripts/19_full_dataset_embeddings.py           full-data MiniLM experiment
scripts/20_full_dataset_qwen.py                 resumable full-data Qwen + LoRA
scripts/21_full_dataset_qwen_context_ablation.py post-training Qwen ablation
scripts/22_collect_final_results.py              combined final results table
scripts/23_qualitative_error_analysis.py         reproducible helped/hurt example analysis
scripts/16_hard_context_ablation.py              semantic hard-negative diagnostic
GPU_RUN.md                                       final GPU execution guide
```

## Main conclusions so far

- **The reply is the main source of signal.** Context-only is weak across methods.
- **More text is not automatically better.** Naive context concatenation hurts TF-IDF.
- **Representation matters.** Separate MiniLM context/reply embeddings yield a small improvement.
- **A sophisticated encoder is not automatically superior.** TF-IDF remains stronger than the frozen MiniLM classifier on the full test set.
- **Context-aware performance can be misleading.** The embedding model is robust to a semantically similar but incorrect context, suggesting semantic compatibility rather than exact conversational understanding.
- The final Qwen experiment is designed specifically to test whether token-level interaction reduces this failure mode.

## References

- Khodak, M., Saunshi, N., & Vodrahalli, K. (2018). *A Large Self-Annotated Corpus for Sarcasm*. LREC.
- Hazarika, D., Poria, S., Gorantla, S., Cambria, E., Zimmermann, R., & Mihalcea, R. (2018). *CASCADE: Contextual Sarcasm Detection in Online Discussion Forums*. COLING.
- Hu, E. J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR.
