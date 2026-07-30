# Paper Tex Consistency Check

Checked against `paper_tex/6.Experimental Evaluation.tex`, subsection `Embedding Model Quality` (`sec:exp:embedding`).

Result: no mismatch found for the requested script scope.

Matches:

- The paper varies embedding model while holding MSMARCO passages and queries fixed.
- The imported runner targets the five models named in the paper: mean-pooled GloVe, mean-pooled FastText, OpenAI ada-002, BGE-M3, and EmbeddingGemma-300M.
- The script expects each embedding-space HDF5 to contain its own exact ground truth via `neighbors`, matching the paper statement that recall is measured within the same embedding space.
- The run scope is FAISS, SIMD on, 24 threads, with the same paper EF ladder and default SAGE calibration knobs used by the other final experiment wrappers.
- The summary script reports iso-recall speedup versus Vanilla, matching the plotted comparison between SAGE and Vanilla in Figure `fig:embedding_quality`.

Intentional exclusions from the old source tree:

- Qwen/VIBE exploratory scripts and generated result artifacts are not part of this paper subsection and were not imported.
- Old date-stamped wrapper names were normalized to stable artifact names.
- The old GloVe-only plotting helper was replaced by `scripts/summarize_msmarco_embedding_model_effects.py`, which handles all five paper models.

Operational caveats:

- The actual HDF5 embedding datasets are not committed. `scripts/preflight_msmarco_embedding_models.py` checks whether the five expected files are present under `SAGE_DATA_DIR`.
- FAISS indexes are also not committed. The runner builds missing `M=32, efConstruction=500` indexes on demand under `SAGE_MSMARCO_EMBEDDING_FAISS_INDEX_ROOT`.
- This check validates script scope and configuration against the tex; it does not re-run the full experiment.
