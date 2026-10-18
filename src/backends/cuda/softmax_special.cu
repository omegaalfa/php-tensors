#include "softmax_special.h"
#include "launch_config.cuh"
#include <math.h>
#include <stdint.h>

template <typename T>
__global__ static void softmax_special_scan_kernel(const T *input,
                                                    size_t total,
                                                    unsigned int *flags)
{
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total)
        return;

    T value = input[index];
    if (isnan(value))
        atomicOr(flags, SOFTMAX_SPECIAL_NAN);
    else if (isinf(value))
        atomicOr(flags, value > (T)0 ? SOFTMAX_SPECIAL_POS_INF
                                     : SOFTMAX_SPECIAL_NEG_INF);
}

template <typename T>
__global__ static void softmax_all_negative_slice_scan_kernel(
    const T *input, size_t outer, size_t inner, int axis_length,
    unsigned int *flags)
{
    size_t slice = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t slices = outer * inner;
    if (slice >= slices || axis_length <= 0)
        return;

    size_t outer_index = slice / inner;
    size_t inner_index = slice % inner;
    size_t base = outer_index * (size_t)axis_length * inner + inner_index;
    for (int j = 0; j < axis_length; j++)
    {
        T value = input[base + (size_t)j * inner];
        if (!(isinf(value) && value < (T)0))
            return;
    }
    atomicOr(flags, SOFTMAX_SPECIAL_ALL_NEG_INF_SLICE);
}

template <typename T>
__device__ static T softmax_exp(T value)
{
    return exp(value);
}

template <typename T>
__global__ static void softmax_special_kernel(const T *input, T *output,
                                               size_t outer, size_t inner,
                                               int axis_length)
{
    size_t slice = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t slices = outer * inner;
    if (slice >= slices || axis_length <= 0)
        return;

    size_t outer_index = slice / inner;
    size_t inner_index = slice % inner;
    size_t base = outer_index * (size_t)axis_length * inner + inner_index;

    int nan_count = 0;
    int positive_infinity_count = 0;
    int negative_infinity_count = 0;
    for (int j = 0; j < axis_length; j++)
    {
        T value = input[base + (size_t)j * inner];
        if (isnan(value))
            nan_count++;
        else if (isinf(value))
        {
            if (value > (T)0)
                positive_infinity_count++;
            else
                negative_infinity_count++;
        }
    }

    if (nan_count != 0)
    {
        for (int j = 0; j < axis_length; j++)
            output[base + (size_t)j * inner] = (T)NAN;
        return;
    }

    if (positive_infinity_count != 0)
    {
        T probability = (T)1 / (T)positive_infinity_count;
        for (int j = 0; j < axis_length; j++)
        {
            T value = input[base + (size_t)j * inner];
            output[base + (size_t)j * inner] = isinf(value) && value > (T)0
                ? probability : (T)0;
        }
        return;
    }

    if (negative_infinity_count == axis_length)
    {
        T probability = (T)1 / (T)axis_length;
        for (int j = 0; j < axis_length; j++)
            output[base + (size_t)j * inner] = probability;
        return;
    }

    T maximum = (T)(-INFINITY);
    for (int j = 0; j < axis_length; j++)
    {
        T value = input[base + (size_t)j * inner];
        if (value != (T)(-INFINITY) && value > maximum)
            maximum = value;
    }

    T sum = (T)0;
    for (int j = 0; j < axis_length; j++)
    {
        T value = input[base + (size_t)j * inner];
        T exponential = value == (T)(-INFINITY)
            ? (T)0 : softmax_exp(value - maximum);
        output[base + (size_t)j * inner] = exponential;
        sum += exponential;
    }

    for (int j = 0; j < axis_length; j++)
        output[base + (size_t)j * inner] /= sum;
}

template <typename T>
static cudaError_t launch_softmax_special_scan_t(const void *input,
                                                 size_t total,
                                                 unsigned int *flags)
{
    softmax_special_scan_kernel<<<cuda_grid_1d(total), 256>>>(
        (const T *)input, total, flags);
    return cudaGetLastError();
}

template <typename T>
static cudaError_t launch_softmax_all_negative_slice_scan_t(
    const void *input, size_t outer, size_t inner, int axis_length,
    unsigned int *flags)
{
    softmax_all_negative_slice_scan_kernel<<<cuda_grid_1d(outer * inner), 256>>>(
        (const T *)input, outer, inner, axis_length, flags);
    return cudaGetLastError();
}

template <typename T>
static cudaError_t launch_softmax_special_t(const void *input, void *output,
                                            size_t outer, size_t inner,
                                            int axis_length)
{
    softmax_special_kernel<<<cuda_grid_1d(outer * inner), 256>>>(
        (const T *)input, (T *)output, outer, inner, axis_length);
    return cudaGetLastError();
}

extern "C" cudaError_t launch_softmax_special_scan(const void *input,
                                                    dtype_t dtype,
                                                    size_t total_elements,
                                                    unsigned int *flags)
{
    if (!input || !flags)
        return cudaErrorInvalidValue;

    unsigned int *device_flags = NULL;
    cudaError_t status = cudaMalloc((void **)&device_flags, sizeof(*device_flags));
    if (status != cudaSuccess)
        return status;
    status = cudaMemset(device_flags, 0, sizeof(*device_flags));
    if (status == cudaSuccess)
    {
        switch (dtype)
        {
        case DTYPE_FLOAT32:
            status = launch_softmax_special_scan_t<float>(input, total_elements, device_flags);
            break;
        case DTYPE_FLOAT64:
            status = launch_softmax_special_scan_t<double>(input, total_elements, device_flags);
            break;
        default:
            status = cudaErrorInvalidValue;
            break;
        }
    }

    if (status == cudaSuccess)
        status = cudaMemcpy(flags, device_flags, sizeof(*flags), cudaMemcpyDeviceToHost);
    cudaError_t free_status = cudaFree(device_flags);
    return status != cudaSuccess ? status : free_status;
}

extern "C" cudaError_t launch_softmax_special(const void *input, void *output,
                                               dtype_t dtype, size_t outer,
                                               size_t inner, int axis_length)
{
    if (!input || !output || !outer || !inner || axis_length <= 0)
        return cudaErrorInvalidValue;
    switch (dtype)
    {
    case DTYPE_FLOAT32:
        return launch_softmax_special_t<float>(input, output, outer, inner, axis_length);
    case DTYPE_FLOAT64:
        return launch_softmax_special_t<double>(input, output, outer, inner, axis_length);
    default:
        return cudaErrorInvalidValue;
    }
}

extern "C" cudaError_t launch_softmax_special_scan_all_negative_slices(
    const void *input, dtype_t dtype, size_t outer, size_t inner,
    int axis_length, unsigned int *flags)
{
    if (!input || !outer || !inner || axis_length <= 0 || !flags)
        return cudaErrorInvalidValue;

    unsigned int *device_flags = NULL;
    cudaError_t status = cudaMalloc((void **)&device_flags, sizeof(*device_flags));
    if (status != cudaSuccess)
        return status;
    status = cudaMemset(device_flags, 0, sizeof(*device_flags));
    if (status == cudaSuccess)
    {
        switch (dtype)
        {
        case DTYPE_FLOAT32:
            status = launch_softmax_all_negative_slice_scan_t<float>(
                input, outer, inner, axis_length, device_flags);
            break;
        case DTYPE_FLOAT64:
            status = launch_softmax_all_negative_slice_scan_t<double>(
                input, outer, inner, axis_length, device_flags);
            break;
        default:
            status = cudaErrorInvalidValue;
            break;
        }
    }
    if (status == cudaSuccess)
        status = cudaMemcpy(flags, device_flags, sizeof(*flags), cudaMemcpyDeviceToHost);
    cudaError_t free_status = cudaFree(device_flags);
    return status != cudaSuccess ? status : free_status;
}
