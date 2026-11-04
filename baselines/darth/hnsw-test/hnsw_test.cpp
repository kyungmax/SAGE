#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <getopt.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <cmath>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>

#include "ComUtil.h"
#include "VectorDataLoader.h"

typedef enum {
  OBSERVATION_DATA_GENERATION = 0,
  DARTH_TESTING = 1,
  NO_EARLY_TERMINATION_TESTING = 2,
  BASELINE_TESTING = 3,
  OPTIMAL_RESULTS_GENERATION = 4,
  LAET_TESTING = 5
} running_mode_t;

char running_mode_str[6][50] = {"Declarative Recall Training Data Generation",
                                "DARTH Testing",
                                "No Early Termination Testing",
                                "Baseline Testing",
                                "Optimal Results Generation",
                                "LAET Testing"};

int main(int argc, char **argv) {
  double t0 = elapsed();

  char *dataset, *output = NULL, *index_filepath = NULL, *predictor_model_path = NULL;

  float target_recall = 0.90;
  int initial_prediction_interval = 1000, min_prediction_interval = 100;

  int logging_interval = 10;

  bool save_index = false;
  running_mode_t mode;
  faiss::idx_t nQ = 100, k = 10;

  int M = 16, efConstruction = 500, efSearch = 500;

  int fixed_amount_of_search = 200;
  float prediction_multiplier = 1.0;

  bool per_prediction_logging = false;
  bool verbose = false;
  int search_threads = omp_get_max_threads();
  bool explicit_threads = false;

  char *dataset_dir_prefix = NULL;

  query_type_t query_type;

  char *noise_perc = "5";

  while (1) {
    static struct option long_options[] = {
        {"dataset", required_argument, 0, '0'},
        {"query-num", required_argument, 0, '1'},
        {"k", required_argument, 0, '2'},
        {"output", required_argument, 0, '3'},
        {"M", required_argument, 0, '4'},
        {"efConstruction", required_argument, 0, '5'},
        {"efSearch", required_argument, 0, '6'},
        {"index-filepath", required_argument, 0, '7'},
        {"save-index", no_argument, 0, '8'},
        {"mode", required_argument, 0, '9'},
        {"target-recall", required_argument, 0, 'a'},
        {"initial-prediction-interval", required_argument, 0, 'b'},
        {"predictor-model-path", required_argument, 0, 'c'},
        {"logging-interval", required_argument, 0, 'd'},
        {"fixed-amount-of-search", required_argument, 0, 'e'},
        {"prediction-multiplier", required_argument, 0, 'f'},
        {"min-prediction-interval", required_argument, 0, 'g'},
        {"per-prediction-logging", no_argument, 0, 'h'},
        {"verbose", no_argument, 0, 'i'},
        {"query-type", required_argument, 0, 'j'},
        {"dataset-dir-prefix", required_argument, 0, 'k'},
        {"gnoise-perc", required_argument, 0, 'l'},
        {"help", no_argument, 0, 'm'},
        {"num-threads", required_argument, 0, 'n'},
        {NULL, 0, NULL, 0}};

    int option_index = 0;
    int c = getopt_long(argc, argv, "", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
      case '0':
        dataset = optarg;
        break;
      case '1':
        sscanf(optarg, "%ld", &nQ);
        break;
      case '2':
        sscanf(optarg, "%ld", &k);
        break;
      case '3':
        output = optarg;
        break;
      case '4':
        sscanf(optarg, "%d", &M);
        break;
      case '5':
        sscanf(optarg, "%d", &efConstruction);
        break;
      case '6':
        sscanf(optarg, "%d", &efSearch);
        break;
      case '7':
        index_filepath = optarg;
        break;
      case '8':
        save_index = true;
        break;
      case '9':
        if (strcmp(optarg, "early-stop-training") == 0) {
          mode = OBSERVATION_DATA_GENERATION;
        } else if (strcmp(optarg, "early-stop-testing") == 0) {
          mode = DARTH_TESTING;
        } else if (strcmp(optarg, "no-early-stop") == 0) {
          mode = NO_EARLY_TERMINATION_TESTING;
        } else if (strcmp(optarg, "naive-early-stop-testing") == 0) {
          mode = BASELINE_TESTING;
        } else if (strcmp(optarg, "optimal-early-stop-testing") == 0) {
          mode = OPTIMAL_RESULTS_GENERATION;
        } else if (strcmp(optarg, "laet-early-stop-testing") == 0) {
          mode = LAET_TESTING;
        } else {
          printf("Unknown running mode: %s\n", optarg);
          exit(EXIT_FAILURE);
        }
        break;
      case 'a':
        sscanf(optarg, "%f", &target_recall);
        break;
      case 'b':
        sscanf(optarg, "%d", &initial_prediction_interval);
        break;
      case 'c':
        predictor_model_path = optarg;
        break;
      case 'd':
        sscanf(optarg, "%d", &logging_interval);
        break;
      case 'e':
        sscanf(optarg, "%d", &fixed_amount_of_search);
        break;
      case 'f':
        sscanf(optarg, "%f", &prediction_multiplier);
        break;
      case 'g':
        sscanf(optarg, "%d", &min_prediction_interval);
        break;
      case 'h':
        per_prediction_logging = true;
        break;
      case 'i':
        verbose = true;
        break;
      case 'j':
        if (strcmp(optarg, "training") == 0) {
          query_type = TRAINING;
        } else if (strcmp(optarg, "validation") == 0) {
          query_type = VALIDATION;
        } else if (strcmp(optarg, "testing") == 0) {
          query_type = TESTING;
        } else if (strcmp(optarg, "noisy-testing") == 0) {
          query_type = NOISY_TESTING;
        } else {
          printf("Unknown query type: %s\n", optarg);
          exit(EXIT_FAILURE);
        }
        break;
      case 'k':
        dataset_dir_prefix = optarg;
        break;
      case 'l':
        // sscanf(optarg, "%f", &gnoise);
        noise_perc = optarg;
        break;
      case 'm':
        printf("This is the driver testing code for the DARTH paper.\n");
        printf("--num-threads N: query workers and offline OpenMP threads "
               "(default: OMP_NUM_THREADS/runtime default).\n");
        exit(EXIT_SUCCESS);
      case 'n': {
        errno = 0;
        char *end = nullptr;
        const long parsed = strtol(optarg, &end, 10);
        if (errno == ERANGE || end == optarg || *end != '\0' ||
            parsed <= 0 || parsed > INT_MAX) {
          fprintf(stderr, "--num-threads must be a positive integer\n");
          return EXIT_FAILURE;
        }
        search_threads = static_cast<int>(parsed);
        explicit_threads = true;
        break;
      }
      default:
        printf("Unknown option: %c\n", c);
        exit(EXIT_FAILURE);
    }
  }

  omp_set_dynamic(0);
  omp_set_num_threads(search_threads);

  if (initial_prediction_interval < min_prediction_interval) {
    printf("Initial prediction interval should be greater than or equal to min prediction interval\n");
    exit(EXIT_FAILURE);
  }

  // print a log with the parameters
  printf(">>Parameters:\n");
  printf("   dataset: %s\n", dataset);
  printf("   nQ: %ld\n", nQ);
  printf("   output: %s\n", output);
  printf("   M: %d, efConstruction: %d, efSearch: %d, k: %ld\n", M, efConstruction, efSearch, k);
  printf("   index_filepath: %s\n", index_filepath);
  printf("   save_index: %s\n", save_index == true ? "yes" : "no");
  printf("   mode: %s\n", running_mode_str[mode]);
  printf("   target_recall: %f\n", target_recall);
  printf("   logging_interval: %d\n", logging_interval);
  printf("   initial_prediction_interval: %d, min_prediction_interval: %d\n", initial_prediction_interval, min_prediction_interval);
  printf("   predictor_model_path: %s\n", predictor_model_path);
  printf("   per_prediction_logging: %s\n", per_prediction_logging == true ? "yes" : "no");
  printf("   query_type: %s\n", query_type_str[query_type]);
  printf("   dataset_dir_prefix: %s\n", dataset_dir_prefix);
  printf("   verbose: %s\n", verbose == true ? "yes" : "no");
  printf("   num_threads: %d (%s)\n", search_threads,
         explicit_threads ? "explicit" : "OpenMP runtime default");
  printf("   gnoise perc: %s\n", noise_perc);
  printf("   [LAET] fixed_amount_of_search: %d, prediction_multiplier: %f\n", fixed_amount_of_search, prediction_multiplier);

  VectorDataLoader vector_dataloader(dataset, query_type, noise_perc, dataset_dir_prefix);
  vector_dataloader.initializeDataMaps();

  // Load database vectors
  size_t d, n;
  float *vecsDB = vector_dataloader.loadDB(&d, &n);
  printf(">>%s DB loaded: d = %ld, n = %ld. Elapsed = %.3fs\n", dataset, d, n, (elapsed() - t0));

  // Load query vectors
  size_t dQ, nQ_all;
  float *vecsQ_all = vector_dataloader.loadQueries(&dQ, &nQ_all);
  assert(dQ == d);

  // Load ground-truth results
  size_t nQ2, k_all;
  int *gt_int_all = vector_dataloader.loadQueriesGroundtruths(&k_all, &nQ2);
  assert(nQ_all == nQ2);

  // Load ground-truth distances
  size_t nQ3, k_all2;
  float *gt_dist_all =
      vector_dataloader.loadQueriesGroundtruthDistances(&k_all2, &nQ3);
  assert(nQ_all == nQ3);

  faiss::idx_t *gt_all = new faiss::idx_t[k_all * nQ_all];
  for (faiss::idx_t i = 0; i < k_all * nQ_all; i++) {
    gt_all[i] = static_cast<faiss::idx_t>(gt_int_all[i]);
  }
  delete[] gt_int_all;

  float *vecsQ;
  faiss::idx_t *gt_k_all, *indicesQ;
  float *gt_k_dist_all;
  get_queries(vecsQ_all, nQ_all, d, gt_all, gt_dist_all, k_all, nQ, &vecsQ, &indicesQ, &gt_k_all, &gt_k_dist_all, false);

  printf(">>k=%ld, k_all=%ld\n", k, k_all);
  assert(k_all >= k);
  faiss::idx_t *gt = new faiss::idx_t[k * nQ];
  float *gt_dist = new float[k * nQ];
  for (faiss::idx_t i = 0; i < nQ; i++) {
    for (faiss::idx_t j = 0; j < k; j++) {
      gt[i * k + j] = gt_k_all[i * k_all + j];
      gt_dist[i * k + j] = gt_k_dist_all[i * k_all + j];
    }
  }

  printf(">>Query Vectors and GT loaded: d = %ld, nQ = %ld, k = %ld, Elapsed = %.3fs\n", dQ, nQ, k, (elapsed() - t0));

  double index_build_start = elapsed();
  faiss::IndexHNSWFlat index(d, M);

  if (save_index) {
    if (index_filepath == NULL) {
      printf(">>Index file path is required to save the index\n");
      exit(EXIT_FAILURE);
    }

    index.hnsw.efConstruction = efConstruction;
    index.hnsw.efSearch = efSearch;
    index.add(n, vecsDB);
    faiss::write_index(&index, index_filepath);

    printf(">>Index saved to %s\n", index_filepath);
  } else {
    if (index_filepath) {
      index = *dynamic_cast<faiss::IndexHNSWFlat *>(faiss::read_index(index_filepath));
      index.hnsw.efSearch = efSearch;
      printf(">>Index loaded from %s\n", index_filepath);
    } else {
      index.hnsw.efConstruction = efConstruction;
      index.hnsw.efSearch = efSearch;
      index.add(n, vecsDB);
    }
  }

  double index_build_time = elapsed() - index_build_start;
  printf(">>Index build time: %.3fs\n", index_build_time);

  faiss::idx_t *I = new faiss::idx_t[nQ * k];
  float *D = new float[nQ * k];

  // Perform the search
  double search_start_time = elapsed();
  faiss::HNSWSearchExecutionStats execution_stats;
  const bool phased_search = mode == DARTH_TESTING ||
      mode == NO_EARLY_TERMINATION_TESTING || mode == BASELINE_TESTING ||
      mode == OPTIMAL_RESULTS_GENERATION;
  double driver_setup_seconds = 0;
  double driver_logging_seconds = 0;

  faiss::DeclarativeRecallDataManager data_manager(D, I, gt, gt_dist, nQ, d, k, vecsQ, output, vecsDB, n, gt_k_all, k_all);
  driver_setup_seconds += elapsed() - search_start_time;
  switch (mode) {
    case OBSERVATION_DATA_GENERATION: {
      faiss::DeclarativeRecallDataCollectorHNSW search_monitor(data_manager, logging_interval);
      search_monitor.init_log_file();
      index.search_declarative_recall_data_generation(nQ, vecsQ, k, D, I, search_monitor);
      search_monitor.close_log_file();
      break;
    }
    case DARTH_TESTING: {
      const double setup_start = elapsed();
      faiss::DARTHPredictorHNSW recall_predictor(data_manager, target_recall, initial_prediction_interval, min_prediction_interval, per_prediction_logging, predictor_model_path);
      recall_predictor.execution_stats = &execution_stats;
      driver_setup_seconds += elapsed() - setup_start;
      const double logging_start = elapsed();
      recall_predictor.init_log_file();
      driver_logging_seconds += elapsed() - logging_start;
      index.search_DARTH(nQ, vecsQ, k, D, I, recall_predictor);
      const double close_start = elapsed();
      recall_predictor.close_log_file();
      driver_logging_seconds += elapsed() - close_start;
      break;
    }
    case NO_EARLY_TERMINATION_TESTING: {
      const double setup_start = elapsed();
      faiss::DeclarativeRecallDataCollectorHNSW searchMonitor(data_manager, INTERVAL_DISABLED_VALUE);
      searchMonitor.execution_stats = &execution_stats;
      driver_setup_seconds += elapsed() - setup_start;
      const double logging_start = elapsed();
      searchMonitor.init_log_file();
      driver_logging_seconds += elapsed() - logging_start;
      index.search_baseline(nQ, vecsQ, k, D, I, searchMonitor);
      const double close_start = elapsed();
      searchMonitor.close_log_file();
      driver_logging_seconds += elapsed() - close_start;
      break;
    }
    case BASELINE_TESTING: {
      const double setup_start = elapsed();
      int distances_to_stop_at = initial_prediction_interval;
      faiss::DeclarativeRecallDataCollectorHNSW search_monitor(data_manager, INTERVAL_DISABLED_VALUE, distances_to_stop_at);
      search_monitor.execution_stats = &execution_stats;
      driver_setup_seconds += elapsed() - setup_start;
      const double logging_start = elapsed();
      search_monitor.init_log_file();
      driver_logging_seconds += elapsed() - logging_start;
      index.search_baseline(nQ, vecsQ, k, D, I, search_monitor);
      const double close_start = elapsed();
      search_monitor.close_log_file();
      driver_logging_seconds += elapsed() - close_start;
      break;
    }
    case OPTIMAL_RESULTS_GENERATION: {
      const double setup_start = elapsed();
      faiss::DeclarativeRecallDataCollectorHNSW search_monitor(data_manager, INTERVAL_WHEN_RECALL_UPDATES_VALUE);
      search_monitor.execution_stats = &execution_stats;
      driver_setup_seconds += elapsed() - setup_start;
      const double logging_start = elapsed();
      search_monitor.init_log_file();
      driver_logging_seconds += elapsed() - logging_start;
      index.search_baseline(nQ, vecsQ, k, D, I, search_monitor);
      const double close_start = elapsed();
      search_monitor.close_log_file();
      driver_logging_seconds += elapsed() - close_start;
      break;
    }
    case LAET_TESTING: {
      faiss::LAETPredictorHNSW laet_predictor(data_manager, fixed_amount_of_search, prediction_multiplier, predictor_model_path);
      laet_predictor.init_log_file();
      index.search_LAET(nQ, vecsQ, k, D, I, laet_predictor);
      laet_predictor.close_log_file();
      break;
    }
    default:
      printf(">>Unknown running mode: %d\n", mode);
      exit(EXIT_FAILURE);
  }

  const double search_and_diagnostics_time = elapsed() - search_start_time;
  const double search_time = phased_search ? execution_stats.search_seconds
                                          : search_and_diagnostics_time;
  if (phased_search) {
    const double setup_time = driver_setup_seconds + execution_stats.setup_seconds;
    const double logging_time = driver_logging_seconds + execution_stats.logging_seconds;
    // Includes result postprocessing, model teardown, and driver overhead.
    const double other_search_time = std::fmax(
        0.0, search_and_diagnostics_time - setup_time - search_time - logging_time);
    printf(">>SearchExecution: requested_threads=%d, worker_count=%d, "
           "active_workers=%d\n", execution_stats.requested_threads,
           execution_stats.worker_count, execution_stats.active_workers);
    printf(">>SearchPhases: SetupAndModelTime: %.9fs, CoreSearchTime: %.9fs, "
           "DiagnosticsAndLoggingTime: %.9fs, "
           "OtherSearchWallTime: %.9fs, "
           "SearchAndDiagnosticsTime: %.9fs (legacy SearchTime scope)\n",
           setup_time, search_time, logging_time, other_search_time,
           search_and_diagnostics_time);
  }

  printf(
      "\n\nIndex[M=%d, efC=%d, efS=%d]IndexTime: %lfs, SearchTime: %lfs, "
      "TotalTime: %lfs, Recall@%ld: %.4f\n",
      M, index.hnsw.efConstruction, index.hnsw.efSearch, index_build_time,
      search_time, (elapsed() - t0), k,
      recall_at_k_avg(gt, I, D, k, nQ, gt_k_all, k_all, verbose));

  delete[] gt_k_all;
  delete[] gt_all;
  delete[] vecsDB;
  delete[] vecsQ;
  delete[] gt;
  delete[] I;
  delete[] D;

  return 0;
}
