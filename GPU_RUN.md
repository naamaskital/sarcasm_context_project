# GPU run guide - final full-dataset experiments

The CPU-side project is complete. The only remaining compute-heavy step is full-dataset Qwen + LoRA training/evaluation.

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

## 3. Train the two final Qwen conditions

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

## 4. Run the post-training context ablation

After `context_plus_comment` has completed:

```bash
python scripts/21_full_dataset_qwen_context_ablation.py
```

This loads the already-trained adapter and evaluates the same full 101,078-example test set under:

- true context
- random wrong context
- same-subreddit wrong context

It also computes paired bootstrap 95% confidence intervals and context-sensitivity counts. No additional Qwen training is performed.

The semantic-nearest hard-negative experiment remains the controlled diagnostic experiment in `scripts/16_hard_context_ablation.py`; it is not repeated over 101k contexts because an exact all-pairs semantic search would change the computational scope substantially.

## 5. Collect the final tables

```bash
python scripts/22_collect_final_results.py
```

The main combined table is written to:

```text
reports/full_dataset/final_results_full_dataset.csv
```

## 6. What to send back for the final report

Copy the console output headed `FINAL` from `scripts/20_full_dataset_qwen.py` and the outputs headed `FINAL METRICS` and `CONTEXT SENSITIVITY` from `scripts/21_full_dataset_qwen_context_ablation.py`.

Everything else in the report can be completed before GPU execution.
