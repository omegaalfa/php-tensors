#include "clip_kernels.h"
#include "cast.cuh"
#include "launch_config.cuh"
#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__device__ T clip_value(T value, T minimum, T maximum)
{
    return value < minimum ? minimum : (value > maximum ? maximum : value);
}

template <>
__device__ float clip_value(float value, float minimum, float maximum)
{
    if (isnan(value)) return value;
    return fminf(fmaxf(value, minimum), maximum);
}

template <>
__device__ double clip_value(double value, double minimum, double maximum)
{
    if (isnan(value)) return value;
    return fmin(fmax(value, minimum), maximum);
}

template <typename T>
static __global__ void clip_kernel(
    const void *input, dtype_t input_dtype, T minimum, T maximum,
    T *output, size_t base_offset, const int *shape,
    const size_t *strides, int ndims, size_t total_size, int is_contiguous)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_size) return;

    size_t offset = index;
    if (!is_contiguous)
    {
        offset = 0;
        size_t remaining = index;
        for (int dimension = ndims - 1; dimension >= 0; dimension--)
        {
            size_t coordinate = remaining % (size_t)shape[dimension];
            remaining /= (size_t)shape[dimension];
            offset += coordinate * strides[dimension];
        }
    }

    T value = fetch_and_cast<T>(input, input_dtype, base_offset + offset);
    output[index] = clip_value(value, minimum, maximum);
}

template <typename T>
static int launch_clip_typed(
    void *input, dtype_t input_dtype, scalar_value_t minimum,
    scalar_value_t maximum, void *output, size_t base_offset,
    const int *device_shape, const size_t *device_strides, int ndims,
    size_t total_size, int is_contiguous)
{
    T minimum_value = convert_union_to_type<T>(minimum);
    T maximum_value = convert_union_to_type<T>(maximum);
    clip_kernel<T><<<cuda_grid_1d(total_size), 256>>>(
        input, input_dtype, minimum_value, maximum_value, (T *)output,
        base_offset, device_shape, device_strides, ndims, total_size,
        is_contiguous);
    return cuda_launch_status(true) == cudaSuccess ? 0 : 1;
}

extern "C" int launch_clip(
    void *input, dtype_t input_dtype, scalar_value_t minimum,
    scalar_value_t maximum, void *output, size_t base_offset,
    const int *device_shape, const size_t *device_strides, int ndims,
    size_t total_size, int is_contiguous)
{
    if (!total_size) return 0;

    switch (input_dtype)
    {
    case DTYPE_FLOAT32:
        return launch_clip_typed<float>(input, input_dtype, minimum, maximum,
            output, base_offset, device_shape, device_strides, ndims,
            total_size, is_contiguous);
    case DTYPE_FLOAT64:
        return launch_clip_typed<double>(input, input_dtype, minimum, maximum,
            output, base_offset, device_shape, device_strides, ndims,
            total_size, is_contiguous);
    case DTYPE_INT8:
        return launch_clip_typed<int8_t>(input, input_dtype, minimum, maximum,
            output, base_offset, device_shape, device_strides, ndims,
            total_size, is_contiguous);
    case DTYPE_INT16:
        return launch_clip_typed<int16_t>(input, input_dtype, minimum, maximum,
            output, base_offset, device_shape, device_strides, ndims,
            total_size, is_contiguous);
    case DTYPE_INT32:
        return launch_clip_typed<int32_t>(input, input_dtype, minimum, maximum,
            output, base_offset, device_shape, device_strides, ndims,
            total_size, is_contiguous);
    case DTYPE_INT64:
        return launch_clip_typed<int64_t>(input, input_dtype, minimum, maximum,
            output, base_offset, device_shape, device_strides, ndims,
            total_size, is_contiguous);
    case DTYPE_UINT8:
        return launch_clip_typed<uint8_t>(input, input_dtype, minimum, maximum,
            output, base_offset, device_shape, device_strides, ndims,
            total_size, is_contiguous);
    case DTYPE_UINT16:
        return launch_clip_typed<uint16_t>(input, input_dtype, minimum, maximum,
            output, base_offset, device_shape, device_strides, ndims,
            total_size, is_contiguous);
    case DTYPE_UINT32:
        return launch_clip_typed<uint32_t>(input, input_dtype, minimum, maximum,
            output, base_offset, device_shape, device_strides, ndims,
            total_size, is_contiguous);
    case DTYPE_UINT64:
        return launch_clip_typed<uint64_t>(input, input_dtype, minimum, maximum,
            output, base_offset, device_shape, device_strides, ndims,
            total_size, is_contiguous);
    case DTYPE_BOOL:
        return launch_clip_typed<bool>(input, input_dtype, minimum, maximum,
            output, base_offset, device_shape, device_strides, ndims,
            total_size, is_contiguous);
    default:
        return 1;
    }
}
