# Paper Tex Consistency Check

Checked against `paper_tex/6.Experimental Evaluation.tex`, subsection `Parameter Sensitivity Analysis` (`sec:exp:ablation`) and the following `Pseudo-Ground-Truth` paragraph/table.

Result: no mismatch found for the requested script scope.

Matches:

- The paper varies five SAGE hyperparameters independently: calibration set size `ncal`, CFR observation window, difficulty tier count `B`, EMA decay `alpha`, and safety margin `g`.
- The imported ablation runner covers exactly those values shown in the paper table: `ncal={100,500,1000}`, window `{[1,13],[4,16]}`, `B={2,4,6}`, `alpha={0.0,0.4,0.8}`, and `g={1,2,3,4}`.
- The runner holds all non-varied settings at the paper defaults: `B=4`, `g=2`, `alpha=0.8`, window `[4,16]`, `ncal=100`, and `tmin_pops=25`.
- The run scope is FAISS, SIMD on, 24 offline/online threads, `M=32`, `efConstruction=500`, `k=10`, and `efSearch=1024`.
- The dataset scope is restricted to `glove-100-angular.hdf5` and `cohere-768-angular.hdf5`, matching the paper ablation table columns (`GloVe100` and `CohereWiki`).
- The pseudo-GT script checks the calibration probes against brute-force exact train-set neighbors, uses FAISS hide-node pseudo-GT at `efSearch=4096`, and recomputes the baseline EF recommendation under pseudo-GT and exact GT.

Intentional exclusions from the old source tree:

- Old final-six, main-eight, GloVe+SpaceV, Cohere+YouTube, hard-stagnation, false-easy, LID-concentration, and plotting/result directories are not imported because they are outside the paper ablation scope requested here.
- Generated CSVs, Markdown summaries, logs, plots, calibration caches, and FAISS indexes are not imported.
- The old broad runners are replaced by a narrow FAISS GloVe/Cohere orchestrator backed by the current local `experiments_scripts/faiss/run_main_qps_latency_sweep.py`.

Operational caveats:

- The scripts validate scope and command construction but do not include result data.
- Full pseudo-GT exact comparison can be expensive on CohereWiki because it brute-force scans the 10M-vector train split for the calibration probes.
- The summary script reports paper-style recall loss as a fraction; the raw runner stores `recall_loss_vs_vanilla_pp` in percent-point units.
