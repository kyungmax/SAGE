# Fresh FAISS target contract

`runs/main/k{10,100}/faiss/targets.json` is written by the new FAISS run only after its
recommended-ef SAGE results are complete. PiP and ABS read this file automatically.
There is no fallback to the archived CSVs.

Top-level fields:

| Field | Value |
| --- | --- |
| `schema_version` | `1` |
| `complete` | `true` only after all configured datasets finish |
| `config_sha256` | SHA-256 of the exact experiment.json bytes |
| `datasets_config_sha256` | SHA-256 of the exact datasets.json bytes |
| `faiss_binary_sha256` | SHA-256 of the active SAGE `_swigfaiss_avx512.so` |
| `datasets` | Ordered list matching the k-specific dataset order (YouTube excluded at k100) |

Each dataset record contains:

| Field | Meaning |
| --- | --- |
| `dataset` | Configured dataset name |
| `recommended_ef` | New probe-only recommendation, minimum max(k, 64) |
| `offline_cap` | Final cap reached during that new calibration |
| `target_hits` | Exact integer top-k hits of SAGE at recommended ef |
| `query_count`, `k` | Entire official query batch and requested k |
| `faiss_index` | Resolved graph identity `{path, bytes, mtime_ns}` from datasets.json |
| `query_sha256` | SHA-256 of raw test vectors converted to contiguous float32, **before normalization** |
| `truth_sha256` | SHA-256 of contiguous int64 `neighbors[:, :k]` |
| `labels_file` | Relative path within this `faiss/` directory to NPZ containing `labels` |
| `labels_sha256` | SHA-256 of contiguous int64 SAGE labels, shape `(query_count, k)` |

Array hashes use `array.tobytes()` in C order. Every baseline process verifies the
file signatures, query/GT hashes and saved SAGE labels, independently recounts the
target hits, and checks graph metadata and sampled stored vectors before tuning.
Target recalls rounded in CSV are never converted back into hits.

The timing scope is the whole batch Python/API call including query preparation
and output allocation, excluding index load, parameter selection, GT evaluation
and file output. All final workers must acquire the configured `timing.lock_file`;
baseline workers use fresh child processes for tuning and final measurement and
enforce the configured CPU mask and NUMA memory policy with `numactl`.

Selection uses evaluation ground truth retrospectively. ABS refines a gamma
bracket to 1e-6 through the configured cap (currently 3.2), checks nearby ticks,
and chooses the smallest tested passing gamma. The retained PiP runner fixes
threshold 0.95 and uses the paper top-k set intersection formula with positive
integer patience (Delta). It uses
the configured Vanilla ef grid (including low extras and the new recommendation),
refines passing patience boundaries, and chooses by separate warmup + 3 selection
repeats. Inversions in observed recall are retained. Neither procedure claims a
global optimum; failure to reach the target is an error, not a matched result.

Historical shared configuration files are preserved to retain the original
SAGE target provenance. PiP overrides only its effective threshold/minimum
patience in the worker and records those values in each new run manifest.
