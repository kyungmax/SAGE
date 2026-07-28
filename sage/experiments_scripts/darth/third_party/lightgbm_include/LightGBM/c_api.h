#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* BoosterHandle;

#define C_API_DTYPE_FLOAT32 0
#define C_API_DTYPE_FLOAT64 1
#define C_API_DTYPE_INT32 2
#define C_API_DTYPE_INT64 3

#define C_API_PREDICT_NORMAL 0
#define C_API_PREDICT_RAW_SCORE 1
#define C_API_PREDICT_LEAF_INDEX 2
#define C_API_PREDICT_CONTRIB 3

const char* LGBM_GetLastError(void);

int LGBM_SetMaxThreads(int num_threads);

int LGBM_BoosterCreateFromModelfile(
    const char* filename,
    int* out_num_iterations,
    BoosterHandle* out);

int LGBM_BoosterPredictForMatSingleRow(
    BoosterHandle handle,
    const void* data,
    int data_type,
    int32_t ncol,
    int is_row_major,
    int predict_type,
    int start_iteration,
    int num_iteration,
    const char* parameter,
    int64_t* out_len,
    double* out_result);

#ifdef __cplusplus
}
#endif
