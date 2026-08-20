# GPU run guide - final full-dataset experiments

The project code now covers the lecturer requirements plus the advanced research extensions. The remaining work is execution and final report integration.

## Final project components

- modern decoder/GPT-style model: Qwen;
- encoder-only comparison: MiniLM;
- Qwen 0.5B vs 1.5B scaling;
- comment-only vs true-context vs random-context controls;
- prompt/input formatting ablation;
- full-data Qwen + LoRA;
- random and same-subreddit context perturbations;
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

CUDA must be `True` before running Qwen.

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

This does not require CUDA and may be run before moving to the GPU machine:

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

The script directly computes:

- `ContextGain = MacroF1(true context) - MacroF1(comment only)`
- `ContextSensitivity = MacroF1(true context) - MacroF1(random context)`

and writes:

```text
reports/full_dataset/qwen_basic_controls/qwen_context_utilization_by_scale.csv
```

For Qwen 0.5B it also compares structured `Previous Reddit message:` / `Reply:` formatting with plain concatenation.

If GPU memory is limited, reduce only inference batch size:

```bash
python scripts/24_qwen_basic_controls.py --batch-size 2
```

## 5. Train the two final full-data Qwen conditions

```bash
python scripts/20_full_dataset_qwen.py --mode comment_only --resume
python scripts/20_full_dataset_qwen.py --mode context_plus_comment --resume
```

The script:

- requires CUDA and fails immediately on CPU;
- trains Qwen2.5-0.5B-Instruct + LoRA for one full epoch over 808,616 training examples;
- evaluates complete validation/test sets;
- checkpoints every 1,000 optimizer steps;
- supports `--resume`;
- saves final adapters and tokenizer;
- saves both predictions and `P(sarcastic)` for behavioral analysis.

If both modes are already complete:

```bash
python scripts/20_full_dataset_qwen.py --mode all --resume --skip-completed
```

## 6. Full-test Qwen context perturbation

```bash
python scripts/21_full_dataset_qwen_context_ablation.py
```

The same trained context adapter is evaluated on all 101,078 IID test examples with:

- true context
- random wrong context
- same-subreddit wrong context

Outputs include Macro-F1 deltas, paired bootstrap 95% CIs, changed-prediction counts, and `P(sarcastic)` under every condition.

## 7. Behavioral interpretability: when does context matter?

After steps 5-6:

```bash
python scripts/26_behavioral_context_interpretability.py
```

It computes per-example probability shifts:

- comment -> true context
- true context -> random context
- true context -> same-subreddit wrong context

and categorizes examples as context-helped, context-hurt, context-irrelevant, context-sensitive, or other. The groups are characterized by comment/context length, semantic similarity, explicit sarcasm markers, and subreddit.

## 8. Refresh qualitative analysis

```bash
python scripts/23_qualitative_error_analysis.py
```

This adds Qwen helped/hurt and perturbation examples to the existing qualitative analysis.

## 9. Collect final tables

```bash
python scripts/22_collect_final_results.py
```

Main table:

```text
reports/full_dataset/final_results_full_dataset.csv
```

## 10. What to send back for final report integration

Send the console sections headed:

- `FINAL UNSEEN-SUBREDDIT RESULTS` and `CONTEXT GAIN ON UNSEEN SUBREDDITS`
- `FINAL BASIC CONTROLS` and `CONTEXT UTILIZATION BY SCALE`
- `FINAL` from full-data Qwen
- `FINAL METRICS` and `CONTEXT SENSITIVITY`
- `WHEN DOES CONTEXT MATTER?`
- `CATEGORY COUNTS`

The report can then be finalized without adding new experiments.
