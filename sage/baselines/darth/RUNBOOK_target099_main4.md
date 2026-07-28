# DARTH / Ada-EF Imported Artifacts

Collected on 2026-07-28 for rerunning and comparing DARTH and Ada-EF experiments.
Artifacts were copied into this directory; original source directories were not removed.

## Layout

- `imported/adaef/final6_target095_20260617/`
  - Source: `../Ada-EF/final6_target095_20260617`
  - Contains Ada-EF final-six summary, manifests, wrapper JSONs, backend JSONs, and skip records.
- `imported/darth/final6_target095_20260617/`
  - Source: `../DARTH/final6_target095_20260617`
  - Contains DARTH final-six summary, manifests, wrapper JSONs, and `online.darth.txt` files.
- `imported/darth/baseline_5datasets_target095_20260603/`
  - Source: `../DARTH/baseline_5datasets_target095_20260603`
  - Contains older 5-dataset DARTH baseline results plus offline cost summary, models, and intervals.
- `imported/darth/spacev100m_target095_20260617/`
  - Source: `../DARTH/spacev100m_target095_20260617`
  - Contains preserved SpaceV 100M DARTH result/log/script artifacts.
- `imported/youtube15m_reference/youtube15m_target095_full_rerun_20260622/`
  - Source: `../youtube15m_target095_full_rerun_20260622`
  - Contains README, scripts, and logs for the YouTube 15M Ada-EF/DARTH rerun.
- `imported/final_implementation_reference/ada-ef/`
  - Source: `../../final_implementation/ada-ef`
  - Contains Ada-EF source, scripts, vendored HNSW headers, README/CMake files, and preserved raw 5-dataset run artifacts.
  - Build outputs from `../../final_implementation/ada-ef/build` were intentionally not copied.
- `imported/final_implementation_reference/darth/scripts/`
  - Source: `../../final_implementation/darth/scripts`
  - Contains general DARTH rerun scripts: `run_paper_offline_fromscratch.py`, `run_training10k.py`, and `prepare_training10k_processed.py`.

## Completion Status

Ada-EF final6 target 0.95:

| Dataset | Status | Recall | Latency ms | Note |
|---|---:|---:|---:|---|
| glove-100 | ok | 0.86641 | 0.9937 |  |
| nytimes | ok | 0.91439 | 1.16717 |  |
| msmarco | ok | 0.990458 | 3.81566 |  |
| cohere-wiki | ok | 0.9783 | 2.35167 | paper label for `cohere-768-angular` |
| sift-100M | skipped |  |  | Ada-EF does not support L2 |
| spacev | skipped |  |  | Ada-EF does not support L2 |

Ada-EF raw final-implementation run `m32_efc500_target095_5datasets_20260615`:

| Dataset | Offline | Online | Recall | QPS | Note |
|---|---:|---:|---:|---:|---|
| nytimes | ok | ok | 0.9308099982 | 447.7366951 | raw `online.json` / `offline.json` copied |
| glove-100 | ok | ok | 0.866369999 | 838.6961742 | raw `online.json` / `offline.json` copied |
| cohere | ok | ok | 0.9782999988 | 377.4597289 | corresponds to cohere-768-angular |
| msmarco | ok | ok | 0.9904584522 | 217.0936066 | raw `online.json` / `offline.json` copied |
| deep-100M | ok | ok | 0.9552699976 | 749.2483726 | not part of final6 label set but preserved in the original run |

DARTH final6 target 0.95:

| Dataset | Status | Recall | Latency ms | Query num |
|---|---:|---:|---:|---:|
| sift-100M | ok | 0.9894 | 9.118096 | 1000 |
| glove-100 | ok | 0.9914 | 12.18097 | 1000 |
| nytimes | ok | 0.9788 | 16.949752 | 1000 |
| msmarco | ok | 0.9985 | 20.150503 | 1000 |
| cohere-wiki | ok | 0.98 | 3.7706 | 100 |
| spacev | ok | 0.9905 | 6.008009 | 1000 |

## Notes

- `../01_main_results/baselines/{adaef,darth}` also contains consolidated copies of the final6 baseline results. It was not copied here to avoid duplicating the same artifacts.
- The YouTube 15M README references large Ada-EF and DARTH result directories under `/home/kyungmin/vectordb/hnsw-playground/index/`, but those referenced directories are not present on this machine. The preserved local material is limited to scripts, logs, README, and non-Ada-EF/DARTH HNSWLib/FAISS outputs in the original final-experiments folder.
- For SpaceV 100M DARTH, the preserved local directory includes result JSON/TXT, logs, and `scripts/run_paper_offline_fromscratch.py`; the original index working directory referenced by its summary is not present.

## DARTH Target 0.99 Main-4 Runbook

Prepared on 2026-07-28 for the target-recall-0.99 rerun set selected from FAISS-recommended efSearch results.

Selected datasets:

| Method | Target recall | Datasets |
|---|---:|---|
| DARTH | 0.99 | agnews, cohere, landmark-nomic, msmarco |
| Ada-EF | 0.99 | cohere, landmark-nomic, msmarco |

DARTH execution wrapper:

- `/home/kyungmin/vectordb/hnsw-playground/trials_on_fixing_search_process/adaptive_efsearch/papers/ours/final_experiments/08_darth_ada-ef/scripts/run_darth_target099_main4.py`

Preserved Python implementation source used by the wrapper:

- `/home/kyungmin/vectordb/hnsw-playground/trials_on_fixing_search_process/adaptive_efsearch/papers/ours/final_experiments/08_darth_ada-ef/imported/final_implementation_reference/darth/scripts/run_paper_offline_fromscratch.py`

Metric-aware DARTH C++ source and binary used by the wrapper:

- Source: `/home/kyungmin/vectordb/hnsw-playground/trials_on_fixing_search_process/adaptive_efsearch/outdated/experiments/darth/benchmarking-darth/hnsw-test/hnsw_test.cpp`
- Binary: `/home/kyungmin/vectordb/hnsw-playground/trials_on_fixing_search_process/adaptive_efsearch/outdated/experiments/darth/benchmarking-darth/build-local/hnsw-test/hnsw_test`
- FAISS shared libraries: `/home/kyungmin/vectordb/hnsw-playground/trials_on_fixing_search_process/adaptive_efsearch/outdated/experiments/darth/benchmarking-darth/build-local/faiss`

Input datasets:

- Root: `/home/kyungmin/vectordb/hnsw-playground/datasets`
- agnews: `/home/kyungmin/vectordb/hnsw-playground/datasets/agnews-mxbai-1024-euclidean.hdf5`
- cohere: `/home/kyungmin/vectordb/hnsw-playground/datasets/cohere-768-angular.hdf5`
- landmark-nomic: `/home/kyungmin/vectordb/hnsw-playground/datasets/landmark-nomic-768-angular.hdf5`
- msmarco: `/home/kyungmin/vectordb/hnsw-playground/datasets/msmarco-v1-openai-ada2-full-ip.hdf5`

Default DARTH output root:

- `/home/kyungmin/vectordb/hnsw-playground/index/darth_m32_efc500_target099_main4_20260728`
- Resolved target on this machine: `/home/smrc/samsung-nvme/kyungmin/index/darth_m32_efc500_target099_main4_20260728`

Planned DARTH settings:

- Threads: 24 for offline/vector/TData/training stages.
- Target recall: 0.99.
- HNSW: M=32, efConstruction=500.
- efSearch: 1000 for DARTH TData generation and online query-time search. LVec generation itself does not use efSearch.
- Online query count: 1000.
- Training queries: 10000 learn, 1000 validation.
- Common FAISS HNSW indexes are reused by default, not rebuilt.

Verified reusable FAISS index root:

- `/home/kyungmin/vectordb/hnsw-playground/index/faiss_m32_efc500_main8_20260707/darth/index`
- Resolved target on this machine: `/home/smrc/samsung-nvme/kyungmin/index/faiss_m32_efc500_main8_20260707/darth/index`

Verified reusable index files:

| Dataset | FAISS type | d | ntotal | metric | Bytes |
|---|---|---:|---:|---|---:|
| agnews | IndexHNSWFlat | 1024 | 769382 | l2 | 3360762154 |
| cohere | IndexHNSWFlat | 768 | 10000000 | ip | 33441292130 |
| landmark-nomic | IndexHNSWFlat | 768 | 760757 | ip | 2544073566 |
| msmarco | IndexHNSWFlat | 1536 | 8841823 | ip | 56730277590 |

Offline cost recording:

- Per-dataset wrapper records: `<run-root>/darth/results/<dataset>/target099.wrapper.json`
- Per-dataset index-build wrapper records: `<run-root>/darth/results/<dataset>/index_build.wrapper.json`
- Consolidated CSV: `<run-root>/darth/results/offline_cost_summary.csv`
- Consolidated JSON: `<run-root>/darth/results/offline_cost_summary.json`
- Run manifest: `<run-root>/RUN_MANIFEST.json`
- Stage logs: `<run-root>/logs/darth/`

Clean-run note:

- The default DARTH output root currently contains an interrupted pre-run with only partial agnews preprocessing artifacts. Use a fresh `--run-root` for the real run, or remove the default output root before rerunning from scratch.
