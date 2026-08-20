# GPU run guide - final focused sarcasm-context project

The project is intentionally focused on one research question: **when and how conversational context improves sarcasm detection**. The code covers the lecturer requirements plus a small set of directly relevant research extensions. Unrelated additions such as agents, RAG, RLHF, or tool use are intentionally excluded.

## Final project components

- classical lexical baseline: TF-IDF + Logistic Regression;
- frozen encoder representation: all-MiniLM-L6-v2;
- direct encoder-only cross-encoder: RoBERTa-base;
- modern decoder/GPT-style model: Qwen2.5;
- Qwen 0.5B vs 1.5B scaling framed as context utilization;
- comment-only, context-only, true-context, and wrong-context controls;
- prompt/input formatting ablation;
- full-data Qwen + LoRA;
- random, same-subreddit, and semantic hard-context perturbations;
- paired bootstrap uncertainty;
- qualitative helped/hurt/error analysis;
- unseen-subreddit generalization;
- behavioral interpretability: when does context matter?

See `RESEARCH_EXTENSIONS.md` for the pre-specified hypotheses and metrics.

## 1. Prepare the GPU environment

```bash
git switch agent/full-dataset
git pull
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build appropriate for the machine if needed. Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

CUDA must be `True` before running GPU experiments.

## 2. Recreate deterministic data protocols

IID full-data split:

```bash
python src/full_dataset_utils.py
```

Expected IID sizes:

- usable total: 1,010,771
- train: 808,616
- validation: 101,077
- test: 101,078

Create the disjoint-community protocol:

```bash
python src/subreddit_generalization_utils.py
```

This groups by `subreddit` and asserts that train, validation, and test subreddit sets do not overlap.

## 3. Unseen-subreddit generalization baseline

This step does not require CUDA:

```bash
python scripts/25_unseen_subreddit_generalization.py
```

It compares comment-only vs context-aware variants for TF-IDF and MiniLM on subreddits absent from training.

Outputs:

```text
reports/subreddit_generalization/unseen_subreddit_metrics.csv
reports/subreddit_generalization/unseen_subreddit_context_gain.csv
```

## 4. Qwen scaling as context utilization + prompt formatting

Run before expensive full-data fine-tuning:

```bash
python scripts/24_qwen_basic_controls.py
```

The fixed balanced held-out sample contains 500 examples per class by default. Both Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct are evaluated under:

- comment only
- true context + comment
- random context + comment

The script computes:

- `ContextGain = MacroF1(true context) - MacroF1(comment only)`
- `ContextSensitivity = MacroF1(true context) - MacroF1(random context)`

For Qwen 0.5B it also compares structured `Previous Reddit message:` / `Reply:` formatting with plain concatenation.

If GPU memory is limited, reduce only inference batch size:

```bash
python scripts/24_qwen_basic_controls.py --batch-size 2
```

## 5. RoBERTa encoder-only cross-encoder

Run the direct encoder-only comparison on the same full IID split:

```bash
python scripts/27_roberta_cross_encoder.py --mode comment_only --resume
python scripts/27_roberta_cross_encoder.py --mode context_only --resume
python scripts/27_roberta_cross_encoder.py --mode context_plus_comment --resume
```

RoBERTa receives context and reply as an actual sentence pair in the combined condition, rather than as two frozen embeddings. It trains for one full epoch and saves probabilities and predictions for the complete test set.

If all modes are already complete:

```bash
python scripts/27_roberta_cross_encoder.py --mode all --resume --skip-completed
```

## 6. Train the three final full-data Qwen controls

Run each separately so progress is recoverable:

```bash
python scripts/20_full_dataset_qwen.py --mode comment_only --resume
python scripts/20_full_dataset_qwen.py --mode context_only --resume
python scripts/20_full_dataset_qwen.py --mode context_plus_comment --resume
```

The script:

- requires CUDA and fails immediately on CPU;
- trains Qwen2.5-0.5B-Instruct + LoRA for one full epoch over 808,616 training examples per condition;
- evaluates complete validation/test sets;
- checkpoints every 1,000 optimizer steps;
- supports `--resume`;
- saves final adapters and tokenizer;
- saves both predictions and `P(sarcastic)` for behavioral analysis.

If all modes are already complete:

```bash
python scripts/20_full_dataset_qwen.py --mode all --resume --skip-completed
```

## 7. Full-test Qwen context perturbation

```bash
python scripts/21_full_dataset_qwen_context_ablation.py
```

The same context-trained adapter is evaluated on all 101,078 IID test examples with:

- true context
- random wrong context
- same-subreddit wrong context

Outputs include Macro-F1 deltas, paired bootstrap 95% CIs, changed-prediction counts, and `P(sarcastic)` under every condition. The semantic-nearest wrong-context condition remains the focused hard-negative diagnostic in `scripts/16_hard_context_ablation.py`.

## 8. Behavioral interpretability: when does context matter?

After Qwen predictions and perturbations exist:

```bash
python scripts/26_behavioral_context_interpretability.py
```

It computes per-example probability shifts:

- comment -> true context
- true context -> random context
- true context -> same-subreddit wrong context

and categorizes examples as context-helped, context-hurt, context-irrelevant, context-sensitive, or other. The groups are characterized by comment/context length, semantic similarity, explicit sarcasm markers, and subreddit.

## 9. Refresh qualitative analysis

```bash
python scripts/23_qualitative_error_analysis.py
```

This adds Qwen helped/hurt and perturbation examples to the existing qualitative analysis.

## 10. Collect final tables

```bash
python scripts/22_collect_final_results.py
```

Main table:

```text
reports/full_dataset/final_results_full_dataset.csv
```

## 11. What to send back for final report integration

Send the console sections headed:

- `FINAL UNSEEN-SUBREDDIT RESULTS` and `CONTEXT GAIN ON UNSEEN SUBREDDITS`
- `FINAL BASIC CONTROLS` and `CONTEXT UTILIZATION BY SCALE`
- `FINAL ROBERTA`
- `FINAL` from full-data Qwen
- `FINAL METRICS` and `CONTEXT SENSITIVITY`
- `WHEN DOES CONTEXT MATTER?`
- `CATEGORY COUNTS`

At that point the report should be finalized by integrating results and discussion only; no unrelated experiments should be added.
