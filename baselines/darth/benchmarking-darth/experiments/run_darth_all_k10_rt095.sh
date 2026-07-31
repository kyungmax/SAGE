#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DARTH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${DARTH_ROOT}/build/hnsw-test/hnsw_test"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SAGE_PROJECT_ROOT:-${SCRIPT_DIR}/../../../..}" && pwd)"
INDEX_ROOT="${INDEX_ROOT:-${DARTH_INDEX_ROOT:-${REPO_ROOT}/index/DARTH}}"
DATASET_ROOT="${DATASET_ROOT:-${DARTH_DATASET_ROOT:-${REPO_ROOT}/datasets/processed/DARTH}}"
MODEL_ROOT="${MODEL_ROOT:-${DARTH_PREDICTOR_ROOT:-${DARTH_ROOT}/predictor_models/darth}}"
TARGET_RECALL="${TARGET_RECALL:-0.95}"
REPEATS="${REPEATS:-3}"
ONLINE_SEARCH_THREADS="${ONLINE_SEARCH_THREADS:-1}"
DATASET_FILTER="${DATASET_FILTER:-}"

# glove100 loader path concatenates without '/', so keep trailing slash to avoid bad path joins.
if [[ "${DATASET_ROOT}" != */ ]]; then
  DATASET_ROOT="${DATASET_ROOT}/"
fi

# Force single-core online search execution for fair QPS comparison.
export OMP_NUM_THREADS="${ONLINE_SEARCH_THREADS}"
export OPENBLAS_NUM_THREADS="${ONLINE_SEARCH_THREADS}"
export MKL_NUM_THREADS="${ONLINE_SEARCH_THREADS}"
export NUMEXPR_NUM_THREADS="${ONLINE_SEARCH_THREADS}"
export VECLIB_MAXIMUM_THREADS="${ONLINE_SEARCH_THREADS}"
export BLIS_NUM_THREADS="${ONLINE_SEARCH_THREADS}"

RUN_TAG="${RUN_TAG:-darth_all_k10_rt095}"
OUT_ROOT="${OUT_ROOT:-${DARTH_ROOT}/experiments/results/${RUN_TAG}}"
LOG_ROOT="${OUT_ROOT}/logs"
CSV_ROOT="${OUT_ROOT}/per_query"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"

mkdir -p "${LOG_ROOT}" "${CSV_ROOT}"

echo "[CONFIG] ONLINE_SEARCH_THREADS=${ONLINE_SEARCH_THREADS}"
echo "[CONFIG] DATASET_ROOT=${DATASET_ROOT}"
if [[ -n "${DATASET_FILTER}" ]]; then
  echo "[CONFIG] DATASET_FILTER=${DATASET_FILTER}"
fi

cat > "${SUMMARY_CSV}" <<'CSV'
dataset,query_num,k,efC,efS,target_recall,ipi,mpi,repeats,avg_recall_mean,p1_recall_mean,p5_recall_mean,search_time_s_mean,qps_mean,exit_code,log_paths,csv_paths
CSV

# dataset|query_num|ipi|mpi|index_path|model_path
declare -a CASES=(
  "agnews-mxbai-1024-euclidean|800|295|59|${INDEX_ROOT}/agnews-mxbai-1024-euclidean/M16_efC200.index|${MODEL_ROOT}/agnews-mxbai-1024-euclidean_M16_efC200_efS2000_s800_k10_nestim100_li2_all_feats.txt"
  "dbpedia-openai-1000k-angular|1000|675|135|${INDEX_ROOT}/dbpedia-openai-1000k-angular/M16_efC200.index|${MODEL_ROOT}/dbpedia-openai-1000k-angular_M16_efC200_efS2000_s1000_k10_nestim100_li2_all_feats.txt"
  "deep-image-96-angular|1000|1204|241|${INDEX_ROOT}/deep-image-96-angular/M16_efC200.index|${MODEL_ROOT}/deep-image-96-angular_M16_efC200_efS2000_s1000_k10_nestim100_li2_all_feats.txt"
  "gist-960-euclidean|800|2581|516|${INDEX_ROOT}/gist-960-euclidean/M16_efC200.index|${MODEL_ROOT}/gist-960-euclidean_M16_efC200_efS2000_s800_k10_nestim100_li2_all_feats.txt"
  "msmarco-v1-openai-ada2-1M-ip|1000|609|122|${INDEX_ROOT}/msmarco-v1-openai-ada2-1M-ip/M16_efC200.index|${MODEL_ROOT}/msmarco-v1-openai-ada2-1M-ip_M16_efC200_efS2000_s1000_k10_nestim100_li2_all_feats.txt"
  "nytimes-256-angular|1000|2110|422|${INDEX_ROOT}/nytimes-256-angular/M16_efC200.index|${MODEL_ROOT}/nytimes-256-angular_M16_efC200_efS2000_s1000_k10_nestim100_li2_all_feats.txt"
  "glove-200-angular|1000|2254|450|${INDEX_ROOT}/glove-200-angular/M16_efC200.index|${MODEL_ROOT}/glove-200-angular_M16_efC200_efS2000_s1000_k10_nestim100_li2_all_feats.txt"
  "glove-100-angular|1000|2462|492|${INDEX_ROOT}/glove-100-angular/M16_efC200.index|${MODEL_ROOT}/glove-100-angular_M16_efC200_efS2000_s1000_k10_nestim100_li2_all_feats.txt"
)

for entry in "${CASES[@]}"; do
  IFS='|' read -r dataset query_num ipi mpi index_path model_path <<< "${entry}"

  if [[ -n "${DATASET_FILTER}" && "${dataset}" != "${DATASET_FILTER}" ]]; then
    continue
  fi

  log_paths=""
  csv_paths=""
  k=10
  efc=200
  efs=2000

  echo "[RUN] dataset=${dataset} query_num=${query_num} k=${k} target_recall=${TARGET_RECALL} repeats=${REPEATS}"

  rc=0
  avg_recall_list=""
  p1_recall_list=""
  p5_recall_list=""
  search_time_list=""
  qps_list=""

  for r in $(seq 1 "${REPEATS}"); do
    log_path="${LOG_ROOT}/${dataset}.r${r}.stdout.log"
    csv_path="${CSV_ROOT}/${dataset}.r${r}.csv"
    log_paths="${log_paths}${log_path};"
    csv_paths="${csv_paths}${csv_path};"

    set +e
    "${BIN}" \
      --dataset "${dataset}" \
      --M 16 --efConstruction "${efc}" --efSearch "${efs}" \
      --query-num "${query_num}" --k "${k}" \
      --mode early-stop-testing \
      --index-filepath "${index_path}" \
      --dataset-dir-prefix "${DATASET_ROOT}" \
      --target-recall "${TARGET_RECALL}" \
      --initial-prediction-interval "${ipi}" \
      --min-prediction-interval "${mpi}" \
      --query-type testing \
      --predictor-model-path "${model_path}" \
      --output "${csv_path}" > "${log_path}" 2>&1
    run_rc=$?
    set -e

    if [[ ${run_rc} -ne 0 ]]; then
      rc=${run_rc}
      break
    fi

    summary_line="$(grep 'SearchTime:' "${log_path}" | tail -n 1 || true)"
    search_time_s="$(echo "${summary_line}" | sed -n 's/.*SearchTime: \([0-9.]*\)s.*/\1/p')"
    avg_recall="$(echo "${summary_line}" | sed -n 's/.*Avg_Recall@[0-9]*: \([0-9.]*\).*/\1/p')"
    p1_recall="$(echo "${summary_line}" | sed -n 's/.*P1_Recall@[0-9]*: \([0-9.]*\).*/\1/p')"
    p5_recall="$(echo "${summary_line}" | sed -n 's/.*P5_Recall@[0-9]*: \([0-9.]*\).*/\1/p')"
    qps="$(awk -v n="${query_num}" -v t="${search_time_s}" 'BEGIN { if (t > 0) printf "%.6f", n / t; else printf "" }')"

    avg_recall_list="${avg_recall_list}${avg_recall} "
    p1_recall_list="${p1_recall_list}${p1_recall} "
    p5_recall_list="${p5_recall_list}${p5_recall} "
    search_time_list="${search_time_list}${search_time_s} "
    qps_list="${qps_list}${qps} "
  done

  avg_recall_mean=""
  p1_recall_mean=""
  p5_recall_mean=""
  search_time_s_mean=""
  qps_mean=""
  if [[ ${rc} -eq 0 ]]; then
    avg_recall_mean="$(awk '{s=0; for(i=1;i<=NF;i++) s+=$i; if(NF>0) printf "%.6f", s/NF;}' <<< "${avg_recall_list}")"
    p1_recall_mean="$(awk '{s=0; for(i=1;i<=NF;i++) s+=$i; if(NF>0) printf "%.6f", s/NF;}' <<< "${p1_recall_list}")"
    p5_recall_mean="$(awk '{s=0; for(i=1;i<=NF;i++) s+=$i; if(NF>0) printf "%.6f", s/NF;}' <<< "${p5_recall_list}")"
    search_time_s_mean="$(awk '{s=0; for(i=1;i<=NF;i++) s+=$i; if(NF>0) printf "%.6f", s/NF;}' <<< "${search_time_list}")"
    qps_mean="$(awk '{s=0; for(i=1;i<=NF;i++) s+=$i; if(NF>0) printf "%.6f", s/NF;}' <<< "${qps_list}")"
  fi

  echo "${dataset},${query_num},${k},${efc},${efs},${TARGET_RECALL},${ipi},${mpi},${REPEATS},${avg_recall_mean},${p1_recall_mean},${p5_recall_mean},${search_time_s_mean},${qps_mean},${rc},${log_paths},${csv_paths}" >> "${SUMMARY_CSV}"

  if [[ ${rc} -eq 0 ]]; then
    echo "[DONE] dataset=${dataset} recall_mean=${avg_recall_mean} search_time_mean_s=${search_time_s_mean} qps_mean=${qps_mean}"
  else
    echo "[FAIL] dataset=${dataset} exit=${rc} logs=${log_paths}"
  fi
done

echo "[SUMMARY] ${SUMMARY_CSV}"
