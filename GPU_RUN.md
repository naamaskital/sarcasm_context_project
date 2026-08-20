# GPU run guide - final focused sarcasm-context project

The project is focused on one research question: **when and how conversational context improves sarcasm detection**. The final workflow follows the scientific narrative rather than a leaderboard:

> baseline -> failure analysis -> targeted fix -> stronger encoders -> architecture-family comparison -> decoder model -> scale -> counterfactual context -> generalization -> interpretability.

## 1. Prepare environment

```bash
git switch agent/full-dataset
git pull
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel --timeout 300 --retries 20
python -m pip install -r requirements.txt --timeout 300 --retries 20
```

The longer timeout/retry settings are intentional because PyTorch/Transformers dependencies and model packages can be large. If a download times out, rerun the same install command; already completed packages do not need to be reinstalled.

Verify CUDA:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

## 2. Recreate deterministic protocols

```bash
python src/full_dataset_utils.py
python src/subreddit_generalization_utils.py
```

Expected IID sizes:
- total usable: 1,010,771
- train: 808,616
- validation: 101,077
- test: 101,078

## 3. Re-run CPU baselines if updated probability files are missing

```bash
python scripts/18_full_dataset_tfidf.py
python scripts/19_full_dataset_embeddings.py
```

## 4. Failure-derived fixes

```bash
python scripts/28_field_aware_tfidf.py
python scripts/29_selective_context_routing.py
```

The first tests lexical dilution from naive context concatenation. The second tests whether context should be used selectively for uncertain replies.

## 5. Unseen-subreddit generalization

```bash
python scripts/25_unseen_subreddit_generalization.py
```

This tests Task-vs-Dataset learning with subreddit-disjoint splits.

## 6. Qwen scaling + prompt-format controls

```bash
python scripts/24_qwen_basic_controls.py
```

This compares Qwen2.5-0.5B and 1.5B under comment-only, true-context, and random-context conditions and computes ContextGain and ContextSensitivity. It also tests structured versus plain prompt formatting.

If memory is limited:

```bash
python scripts/24_qwen_basic_controls.py --batch-size 2
```

## 7. Encoder-only progression

### BERT
```bash
python scripts/30_bert_cross_encoder.py --mode all --resume --skip-completed
```

### RoBERTa
```bash
python scripts/27_roberta_cross_encoder.py --mode all --resume --skip-completed
```

### DeBERTa
```bash
python scripts/31_deberta_cross_encoder.py --mode all --resume --skip-completed
```

All three use the original class-presentation conditions:
- comment only
- context only
- context + comment

This tests whether stronger joint encoder interaction improves context utilization.

## 8. Encoder-Decoder comparison

### FLAN-T5-base
```bash
python scripts/32_flan_t5_encoder_decoder.py --mode all --resume --skip-completed
```

FLAN-T5 uses the same three input conditions and completes the course-aligned comparison between Encoder-only, Encoder-Decoder, and Decoder-only Transformer families.

## 9. Full-data Qwen2.5-0.5B + LoRA

```bash
python scripts/20_full_dataset_qwen.py --mode all --resume --skip-completed
```

This trains comment-only, context-only, and context+comment for one full epoch each, with checkpoints and probability outputs.

## 10. Qwen counterfactual context evaluation

```bash
python scripts/21_full_dataset_qwen_context_ablation.py
```

Evaluates true, random wrong, and same-subreddit wrong context with paired bootstrap confidence intervals and prediction-change statistics. Semantic-similar wrong context remains the focused diagnostic in `scripts/16_hard_context_ablation.py`.

## 11. Behavioral interpretability

```bash
python scripts/26_behavioral_context_interpretability.py
```

Analyzes probability shifts and characterizes context-helped, context-hurt, context-irrelevant, and context-sensitive examples.

## 12. Refresh qualitative examples

```bash
python scripts/23_qualitative_error_analysis.py
```

## 13. Collect final tables

```bash
python scripts/22_collect_final_results.py
```

Outputs:
```text
reports/full_dataset/final_results_full_dataset.csv
reports/full_dataset/final_auxiliary_results.csv
```

The main table now includes TF-IDF, MiniLM, BERT, RoBERTa, DeBERTa, FLAN-T5, and Qwen. The auxiliary table contains hypothesis-driven follow-ups such as context routing, hard negatives, scaling, and generalization.

## 14. What to send back for final report integration

Send the console sections for:
- updated TF-IDF/MiniLM if rerun
- field-aware TF-IDF
- selective context routing
- unseen-subreddit results
- Qwen basic controls + context utilization by scale
- `FINAL BERT`
- `FINAL ROBERTA`
- `FINAL DEBERTA`
- `FINAL FLAN-T5`
- full-data Qwen `FINAL`
- Qwen `FINAL METRICS` + `CONTEXT SENSITIVITY`
- `WHEN DOES CONTEXT MATTER?`
- `CATEGORY COUNTS`

After that, the remaining work is report integration and interpretation, not adding unrelated experiments.