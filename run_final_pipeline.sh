#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="reports/final_run_logs"
mkdir -p "$LOG_DIR"

run_step() {
  local name="$1"
  shift
  echo
  echo "======================================================================"
  echo "STEP: $name"
  echo "COMMAND: $*"
  echo "======================================================================"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    raise SystemExit('CUDA is not available. Stop before the final run.')
PY

# Deterministic data protocols
run_step 01_full_dataset_split python src/full_dataset_utils.py
run_step 02_subreddit_split python src/subreddit_generalization_utils.py

# Baselines and failure-driven follow-ups
run_step 03_tfidf python scripts/18_full_dataset_tfidf.py
run_step 04_minilm python scripts/19_full_dataset_embeddings.py
run_step 05_field_aware_tfidf python scripts/28_field_aware_tfidf.py
run_step 06_selective_context_routing python scripts/29_selective_context_routing.py
run_step 07_unseen_subreddit python scripts/25_unseen_subreddit_generalization.py

# Controlled scaling / prompting before fine-tuning
run_step 08_qwen_basic_controls python scripts/24_qwen_basic_controls.py --batch-size 4

# Transformer architecture comparison
run_step 09_bert python scripts/30_bert_cross_encoder.py --mode all --resume
run_step 10_roberta python scripts/27_roberta_cross_encoder.py --mode all --resume
run_step 11_deberta python scripts/31_deberta_cross_encoder.py --mode all --resume
run_step 12_flan_t5 python scripts/32_flan_t5_encoder_decoder.py --mode all --resume

# Full-data decoder + PEFT
run_step 13_qwen_full python scripts/20_full_dataset_qwen.py --mode all --resume
run_step 14_qwen_context_ablation python scripts/21_full_dataset_qwen_context_ablation.py

# Interpretability and qualitative synthesis
run_step 15_behavioral_interpretability python scripts/26_behavioral_context_interpretability.py
run_step 16_qualitative_analysis python scripts/23_qualitative_error_analysis.py
run_step 17_collect_results python scripts/22_collect_final_results.py

echo
echo "======================================================================"
echo "FINAL PIPELINE COMPLETED SUCCESSFULLY"
echo "Logs: $LOG_DIR"
echo "Main results: reports/full_dataset/final_results_full_dataset.csv"
echo "Auxiliary results: reports/full_dataset/final_auxiliary_results.csv"
echo "======================================================================"
