# GPU run guide - final full-dataset experiments

The CPU-side project is complete. The remaining compute-heavy work is the final Qwen execution plus the mandatory basic controls requested in the project proposal/lecturer feedback.

## Final requirements covered by the GPU run

The final project must include all of the following:

- modern decoder/GPT-style model: Qwen;
- encoder-only comparison: MiniLM is already complete on the full dataset;
- at least two Qwen sizes in a basic controlled experiment: 0.5B and 1.5B;
- comment-only versus context+comment;
- prompt/input formatting control;
- full-data Qwen + LoRA result;
- post-training random and same-subreddit context ablations;
- paired bootstrap uncertainty and context-sensitivity analysis;
- qualitative helped/hurt/error analysis.

## 1. Prepare the environment

```bash
git switch agent/full-dataset
git pull
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build appropriate for the GPU machine if the default installation is CPU-only. Verify before training:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

The last two lines must show `True` and a GPU name.

## 2. Recreate the fixed full-dataset split

```bash
python src/full_dataset_utils.py
```

Expected cleaned corpus and fixed split sizes:

- usable total: 1,010,771
- train: 808,616
- validation: 101,077
- test: 101,078

The split is deterministic (`SEED=42`).

## 3. Mandatory basic controls: model size + prompt formatting

Run this before the expensive full-data fine-tuning:

```bash
python scripts/24_qwen_basic_controls.py
```

Default evaluation uses a fixed balanced subset of the full held-out test split: 500 examples per class (1,000 total), with identical 2-shot-per-class demonstrations.

It performs:

### Qwen size comparison

- Qwen2.5-0.5B-Instruct, comment only
- Qwen2.5-0.5B-Instruct, structured context + reply
- Qwen2.5-1.5B-Instruct, comment only
- Qwen2.5-1.5B-Instruct, structured context + reply

This directly satisfies the lecturer request to test at least two modern GPT/decoder model sizes in the basic experiment.

### Prompt-formatting control

For Qwen2.5-0.5B, the same context+reply examples are also evaluated using plain concatenation instead of explicit `Previous Reddit message:` / `Reply:` fields.

Outputs:

```text
reports/full_dataset/qwen_basic_controls/qwen_basic_controls_metrics.csv
reports/full_dataset/qwen_basic_controls/qwen_basic_controls_predictions.parquet
reports/full_dataset/qwen_basic_controls/qwen_basic_controls_summary.txt
```

If GPU memory is limited, reduce only the inference batch size, not the evaluation sample:

```bash
python scripts/24_qwen_basic_controls.py --batch-size 2
```

## 4. Train the two final full-data Qwen conditions

Run each condition separately so a failure does not lose progress from the other one:

```bash
python scripts/20_full_dataset_qwen.py --mode comment_only --resume
python scripts/20_full_dataset_qwen.py --mode context_plus_comment --resume
```

The script:

- requires CUDA and fails immediately on CPU;
- trains Qwen2.5-0.5B-Instruct + LoRA for one epoch over all 808,616 training examples;
- evaluates on the full validation and test sets;
- writes checkpoints every 1,000 optimizer steps;
- can resume from the latest checkpoint with `--resume`;
- saves the final LoRA adapter and tokenizer under `models/full_dataset_qwen/<mode>/final_adapter/`;
- saves exact metrics under `reports/full_dataset/qwen/`.

If both modes are already complete, this command safely skips them:

```bash
python scripts/20_full_dataset_qwen.py --mode all --resume --skip-completed
```

## 5. Run the post-training context ablation

After `context_plus_comment` has completed:

```bash
python scripts/21_full_dataset_qwen_context_ablation.py
```

This loads the already-trained adapter and evaluates the same full 101,078-example test set under:

- true context
- random wrong context
- same-subreddit wrong context

It also computes paired bootstrap 95% confidence intervals and context-sensitivity counts. No additional Qwen training is performed.

The semantic-nearest hard-negative condition remains the focused diagnostic experiment in `scripts/16_hard_context_ablation.py`; it is not repeated over 101k contexts because an exact all-pairs semantic search would change the computational scope substantially.

## 6. Refresh qualitative analysis

After Qwen predictions exist, rerun:

```bash
python scripts/23_qualitative_error_analysis.py
```

This automatically adds Qwen context-helped/context-hurt and Qwen perturbation examples to the existing qualitative analysis.

## 7. Collect the final tables

```bash
python scripts/22_collect_final_results.py
```

The main combined table is written to:

```text
reports/full_dataset/final_results_full_dataset.csv
```

## 8. What to send back for the final report

Send the console sections headed:

- `FINAL BASIC CONTROLS` from `scripts/24_qwen_basic_controls.py`
- `FINAL` from `scripts/20_full_dataset_qwen.py`
- `FINAL METRICS` and `CONTEXT SENSITIVITY` from `scripts/21_full_dataset_qwen_context_ablation.py`
- `CATEGORY COUNTS` from the rerun of `scripts/23_qualitative_error_analysis.py`

Everything else in the report can remain complete before GPU execution.
