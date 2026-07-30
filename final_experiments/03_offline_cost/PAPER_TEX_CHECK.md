# Paper Experiment-Section Check

Checked against `paper_tex/6.Experimental Evaluation.tex`, subsection `Offline Calibration Cost`.

Matched settings:

- paper uses 24 threads; wrappers pass `--offline-num-threads 24` and export thread env vars to 24.
- paper setup states FAISS and hnswlib are compiled with SIMD enabled; FAISS wrapper forces `FAISS_OPT_LEVEL=AVX512`, hnswlib wrapper points at the compiled extension root.
- paper table covers eight datasets; the runner defaults to the same eight in table order.
- paper excludes HNSW index build time; the runner records dataset/index load time separately and keeps paper-facing timing in calibration-only columns.
- paper uses three calibration steps: `Samp.`, `Select`, and `Eval.`; the runner writes these as `step1_lid_sampling_wall_s`, `step2_pre_evaluation_wall_s`, and `step3_eval_wall_s`, plus total `offline_calibration_wall_s`.
- paper defaults are preserved: `M=32`, `efConstruction=500`, `ncal=100`, sampled 10,000-node LID pool, pseudo-GT `efSearch=4096`, EF sweep `64..1024`, `B=4`, `g=2`, classify window `[4,16]`, EMA `alpha=0.8`.

Intentional differences from old source bundles:

- Generated CSVs, logs, manifests, caches, and previous result directories are not imported.
- The old source had one confusingly named hnswlib-only runner that also had a FAISS branch. This artifact exposes a backend-neutral runner plus explicit FAISS/hnswlib wrappers.
- The runner defaults to three repeats and writes a median CSV because the paper caption says values are the median of three runs.
