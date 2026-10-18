#ifndef PHPTENSORS_SOFTMAX_SPECIAL_H
#define PHPTENSORS_SOFTMAX_SPECIAL_H

#include <cuda_runtime.h>
#include "../../data_types.h"

#define SOFTMAX_SPECIAL_NAN 1u
#define SOFTMAX_SPECIAL_POS_INF 2u
#define SOFTMAX_SPECIAL_NEG_INF 4u
#define SOFTMAX_SPECIAL_ALL_NEG_INF_SLICE 8u

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t launch_softmax_special_scan(const void *input, dtype_t dtype,
                                        size_t total_elements,
                                        unsigned int *flags);
cudaError_t launch_softmax_special_scan_all_negative_slices(
    const void *input, dtype_t dtype, size_t outer, size_t inner,
    int axis_length, unsigned int *flags);
cudaError_t launch_softmax_special(const void *input, void *output,
                                   dtype_t dtype, size_t outer,
                                   size_t inner, int axis_length);

#ifdef __cplusplus
}
#endif

#endif
