#ifndef CLIP_KERNELS_H
#define CLIP_KERNELS_H

#include "../../data_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int launch_clip(
    void *input, dtype_t input_dtype, scalar_value_t minimum,
    scalar_value_t maximum, void *output, size_t base_offset,
    const int *device_shape, const size_t *device_strides, int ndims,
    size_t total_size, int is_contiguous);

#ifdef __cplusplus
}
#endif

#endif
