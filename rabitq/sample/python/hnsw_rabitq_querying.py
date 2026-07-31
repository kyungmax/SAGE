import argparse
import numpy as np
from time import time
from rabitqlib import HnswIndex
from utils import read_fvecs, read_ivecs, compute_recall, l2_normalize_rows

# ──────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────
TOPK        = 10              # top-k results
NUM_THREADS = 1               #number of threads
EFS         = [64, 128, 256, 512, 1024]
TEST_ROUNDS = 3
# ──────────────────────────────────────────────


def main(args=None) -> None:
    # 1. Load queries and ground truth
    queries = read_fvecs(args.query_file, hdf5_key=args.query_hdf5_key)
    if args.normalize:
        print("L2-normalizing queries")
        queries = l2_normalize_rows(queries)
    gt      = read_ivecs(args.gt_file, hdf5_key=args.gt_hdf5_key)
    nq      = queries.shape[0]
    print(f"Queries: {queries.shape}, GT: {gt.shape}")

    # 2. Load index
    idx = HnswIndex.load(args.index_file)
    print(f"Index loaded — dim={idx.dim}")
    print(f"TopK: {args.topk}")

    print("\nsearch start >.....\n")

    efs = args.efs if args.efs is not None else EFS
    all_qps    = np.zeros((args.test_rounds, len(efs)))
    all_recall = np.zeros((args.test_rounds, len(efs)))
    stats_by_ef = {}

    for i_probe, ef in enumerate(efs):
        for r in range(args.test_rounds):
            t0 = time()
            if args.adaptive_light:
                ids, _, stats = idx.search_adaptive_light_with_stats(
                    queries,
                    k=args.topk,
                    ef_init=ef,
                    enable_stop=args.enable_stop,
                    num_threads=args.num_threads,
                    early_stop_ratio=args.early_stop_ratio,
                    tmin_pops=args.tmin_pops,
                    super_easy_gamma_ratio=args.super_easy_gamma_ratio,
                    mid_easy_upper_gamma_ratio=args.mid_easy_upper_gamma_ratio,
                    ef_max=args.ef_max,
                    paper_bucket_mode=args.paper_bucket_mode,
                    paper_bucket_count=args.paper_bucket_count,
                    bucket_gamma_ratios=args.bucket_gamma_ratios,
                )
                if r == 0:
                    stats_by_ef[ef] = {key: np.asarray(value) for key, value in stats.items()}
            else:
                ids, _ = idx.search(queries, k=args.topk, ef=ef, num_threads=args.num_threads)
            elapsed = time() - t0  # seconds

            qps    = nq / elapsed
            recall = compute_recall(ids, gt, args.topk)

            all_qps[r, i_probe]    = qps
            all_recall[r, i_probe] = recall

    avg_qps    = all_qps.mean(axis=0)
    avg_recall = all_recall.mean(axis=0)

    # 3. Print results table
    mode = "adaptive-light" if args.adaptive_light else "regular"
    print(f"Mode: {mode}")
    print(f"{'EF':<8}{'QPS':<14}{'Recall'}")
    print("-" * 35)
    for i, ef in enumerate(efs):
        print(f"{ef:<8}{avg_qps[i]:<14.1f}{avg_recall[i]:<12.4f}")

    if args.adaptive_light:
        print("\nAdaptive stats")
        for ef in efs:
            stats = stats_by_ef.get(ef)
            if stats is None:
                continue
            classified = stats["classified"].astype(bool)
            easy = stats["easy_query"].astype(bool)
            super_easy = stats.get("super_easy_query", np.zeros(nq, dtype=bool)).astype(bool)
            mid_easy = stats.get("mid_easy_query", np.zeros(nq, dtype=bool)).astype(bool)
            shrunk = stats["ef_shrunk"].astype(bool)
            stopped = stats["early_stopped"].astype(bool)
            effective = stats["effective_ef"]
            full_pops = stats["full_pop_count"]
            base_bin = stats.get("base_bin_est_count")
            base_full = stats.get("base_full_est_count")
            cfr_mean = stats["classify_cfr_mean"]
            finite_cfr = cfr_mean[np.isfinite(cfr_mean)]
            cfr_avg = float(finite_cfr.mean()) if finite_cfr.size else float("nan")
            full_ratio = (float(base_full.sum()) / float(base_bin.sum())) if base_bin is not None and base_bin.sum() else float("nan")
            print(
                f"EF {ef}: effective_ef mean/min/max="
                f"{effective.mean():.1f}/{effective.min()}/{effective.max()}, "
                f"classified={classified.sum()}/{nq}, easy={easy.sum()}/{nq}, "
                f"super_easy={super_easy.sum()}/{nq}, mid_easy={mid_easy.sum()}/{nq}, "
                f"shrunk={shrunk.sum()}/{nq}, early_stopped={stopped.sum()}/{nq}, "
                f"full_pops_mean={full_pops.mean():.1f}, classify_cfr_mean={cfr_avg:.4f}, "
                f"base_full_est_ratio={full_ratio:.4f}"
            )

        if args.stats_out:
            rows = []
            for ef in efs:
                stats = stats_by_ef.get(ef)
                if stats is None:
                    continue
                for qid in range(nq):
                    rows.append([
                        ef,
                        qid,
                        int(stats["initial_ef"][qid]),
                        int(stats["effective_ef"][qid]),
                        int(stats["full_pop_count"][qid]),
                        int(stats.get("base_bin_est_count", np.zeros(nq, dtype=np.int64))[qid]),
                        int(stats.get("base_full_est_count", np.zeros(nq, dtype=np.int64))[qid]),
                        int(stats["classified"][qid]),
                        int(stats["easy_query"][qid]),
                        int(stats.get("super_easy_query", np.zeros(nq, dtype=bool))[qid]),
                        int(stats.get("mid_easy_query", np.zeros(nq, dtype=bool))[qid]),
                        int(stats["ef_shrunk"][qid]),
                        int(stats["early_stopped"][qid]),
                        float(stats["classify_cfr_mean"][qid]),
                    ])
            header = "ef_init,query_id,initial_ef,effective_ef,full_pop_count,base_bin_est_count,base_full_est_count,classified,easy_query,super_easy_query,mid_easy_query,ef_shrunk,early_stopped,classify_cfr_mean"
            np.savetxt(args.stats_out, np.asarray(rows, dtype=object), delimiter=",", header=header, comments="", fmt="%s")
            print(f"Per-query adaptive stats saved -> {args.stats_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RaBitQ HNSW Querying")

    parser.add_argument("index_file", type=str, help="Path to the HNSW index file")
    parser.add_argument("query_file", type=str,  help="Path to the query file")
    parser.add_argument("gt_file", type=str, help="Path to the ground truth file")
    parser.add_argument("--topk", dest="topk", type=int, metavar="INT", default=TOPK, help="Number of top results to retrieve")
    parser.add_argument("--num-threads", dest="num_threads", type=int, metavar="INT", default=NUM_THREADS, help="Number of threads for searching")
    parser.add_argument("--test-rounds", dest="test_rounds", type=int, metavar="INT", default=TEST_ROUNDS, help="Number of test rounds for averaging")
    parser.add_argument("--efs", dest="efs", type=int, nargs="+", default=None, help="EF values to benchmark; defaults to [128] for adaptive-light and a sweep otherwise")
    parser.add_argument("--adaptive-light", dest="adaptive_light", action="store_true", help="Use CFR adaptive-light search")
    parser.add_argument("--disable-stop", dest="enable_stop", action="store_false", default=True, help="Disable hard-query stagnation stop")
    parser.add_argument("--early-stop-ratio", dest="early_stop_ratio", type=float, default=0.6, help="CFR threshold for easy-query classification")
    parser.add_argument("--tmin-pops", dest="tmin_pops", type=int, default=25, help="Minimum full-queue pops before hard-query stop")
    parser.add_argument("--ef-max", dest="ef_max", type=int, default=1024, help="Maximum effective ef for adaptive-light")
    parser.add_argument("--super-easy-gamma-ratio", dest="super_easy_gamma_ratio", type=float, default=np.nan, help="Optional super-easy bucket threshold")
    parser.add_argument("--mid-easy-upper-gamma-ratio", dest="mid_easy_upper_gamma_ratio", type=float, default=np.nan, help="Optional mid-easy bucket threshold")
    parser.add_argument("--paper-bucket-mode", dest="paper_bucket_mode", action="store_true", help="Use paper-style bucket ef routing")
    parser.add_argument("--paper-bucket-count", dest="paper_bucket_count", type=int, default=4, help="Number of paper buckets, 2 to 8")
    parser.add_argument("--bucket-gamma-ratios", dest="bucket_gamma_ratios", type=float, nargs="*", default=[], help="Monotone bucket gamma ratios for paper bucket mode")
    parser.add_argument("--stats-out", dest="stats_out", type=str, default=None, help="Optional CSV path for per-query adaptive stats")
    parser.add_argument("--query-hdf5-key", dest="query_hdf5_key", type=str, default="test", help="HDF5 dataset key for queries")
    parser.add_argument("--gt-hdf5-key", dest="gt_hdf5_key", type=str, default="neighbors", help="HDF5 dataset key for ground truth")
    parser.add_argument("--normalize", dest="normalize", action="store_true", help="L2-normalize queries before search")
    args = parser.parse_args()
    
    main(args)
