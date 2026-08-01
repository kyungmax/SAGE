#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
PY="${SAGE_PYTHON:-python3}"

DATASETS="${SAGE_DRILLDOWN_DATASETS:-glove-100-angular.hdf5,nytimes-256-angular.hdf5,msmarco-v1-openai-ada2-full-ip.hdf5,msspacev-100M-i8-euclidean.hdf5,cohere-768-angular.hdf5,youtube-15M-angular.hdf5,agnews-mxbai-1024-euclidean.hdf5,landmark-nomic-768-angular.hdf5}"
ANALYSIS_DATASETS="${SAGE_ANALYSIS_DATASETS:-glove-100-angular,nytimes-256-angular,msmarco-v1-openai-ada2-full-ip,msspacev-100M-i8-euclidean,cohere-768-angular,youtube-15M-angular,agnews-mxbai-1024-euclidean,landmark-nomic-768-angular}"
EFS="${SAGE_DRILLDOWN_EFS:-1024}"
CAL_EFS="64,80,96,128,160,192,256,320,384,512,640,768,896,1024"
RUN_ID="${SAGE_RUN_ID:-drilldown_faiss_SIMD_on_main8_24t}"

DATA_DIR="${SAGE_DATA_DIR:-$REPO_ROOT/datasets}"
INDEX_DIR="${SAGE_INDEX_DIR:-$REPO_ROOT/index}"
FAISS_PYTHON="${FAISS_PYTHON_PATH:-$REPO_ROOT/faiss/build_sage_avx512/faiss/python}"
FAISS_INDEX="${SAGE_FAISS_INDEX_ROOT:-${FAISS_INDEX_ROOT:-$INDEX_DIR/faiss_m32_efc500_main8/index}}"

DIFF_DIR="$ROOT/$RUN_ID/difficulty_exactgt_24t"
POLICY_DIR="$ROOT/$RUN_ID/policy_run"
HARD_DIR="$ROOT/$RUN_ID/hard_loss_querywise_exactgt_24t"
LARGE_DIR="$ROOT/$RUN_ID/large_false_easy_drop_analysis"
LOG_DIR="$ROOT/$RUN_ID/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/faiss_SIMD_on_24t.log"
exec > >(tee -a "$LOG_FILE") 2>&1

export FAISS_OPT_LEVEL="${FAISS_OPT_LEVEL:-AVX512}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-24}"

cd "$ROOT"

echo "[START] $(date -Is) backend=faiss SIMD=${FAISS_OPT_LEVEL} threads=24 efs=${EFS}"
echo "[LOG] $LOG_FILE"
echo "[RUN_ID] $RUN_ID"

echo "[STEP] difficulty_exactgt_24t"
"$PY" scripts/final8_faiss_difficulty_drilldown_24t.py \
  --backend faiss \
  --faiss-python-path "$FAISS_PYTHON" \
  --faiss-index-root "$FAISS_INDEX" \
  --datasets "$DATASETS" \
  --base-path "$DATA_DIR" \
  --index-dir "$INDEX_DIR" \
  --output-dir "$DIFF_DIR" \
  --run-root "$POLICY_DIR" \
  --efs "$EFS" \
  --calibration-efs "$CAL_EFS" \
  --eval-gt-source exact \
  --offline-num-threads 24 \
  --online-num-threads 24 \
  --warmup-runs 1 \
  --measured-runs 3 \
  --param-m 32 \
  --ef-construction 500 \
  --num-calibration-queries 100 \
  --classify-start 4 \
  --classify-end 16 \
  --cfr-ema-decay 0.8 \
  --pair-gap 2 \
  --tmin-pops 25 \
  --mixed-threshold-mode paper_floor_half \
  --mixed-bucket-count 4

echo "[STEP] hard_loss_querywise_exactgt_24t"
"$PY" scripts/final8_faiss_hard_loss_querywise_replay_24t.py \
  --backend faiss \
  --faiss-python-path "$FAISS_PYTHON" \
  --faiss-index-root "$FAISS_INDEX" \
  --run-root "$POLICY_DIR" \
  --query-groups "$DIFF_DIR/query_groups.csv" \
  --output-dir "$HARD_DIR" \
  --datasets "$DATASETS" \
  --base-path "$DATA_DIR" \
  --index-dir "$INDEX_DIR" \
  --efs "$EFS" \
  --eval-gt-source exact \
  --num-threads 24 \
  --workers 24 \
  --batch-size 512 \
  --cfr-batch-size 2048 \
  --param-m 32 \
  --ef-construction 500 \
  --tmin-pops 25 \
  --classify-start 4 \
  --classify-end 16 \
  --cfr-ema-decay 0.8

echo "[STEP] large_false_easy_drop_analysis"
"$PY" scripts/analyze_large_false_easy_drops.py \
  --querywise "$HARD_DIR/hard_loss_querywise.csv" \
  --query-groups "$DIFF_DIR/query_groups.csv" \
  --output-dir "$LARGE_DIR" \
  --datasets "$ANALYSIS_DATASETS" \
  --output-prefix main8_SIMD_on \
  --min-drop 0.4 \
  --k 10

echo "[END] $(date -Is) backend=faiss"
