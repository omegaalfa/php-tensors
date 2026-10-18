#ifndef PHPTENSORS_HOST_ARRAY_UNSUPPORTED_H
#define PHPTENSORS_HOST_ARRAY_UNSUPPORTED_H

#include "php.h"

#define HOST_ARRAY_UNSUPPORTED_RETURN \
    ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_tensor, 0, 0, PhpTensors\\HostArray, 0) \
    ZEND_END_ARG_INFO()

HOST_ARRAY_UNSUPPORTED_RETURN

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_from_file, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, path, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_from_npy, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, path, IS_STRING, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_where, 0, 3, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_INFO(0, condition, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_INFO(0, x, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_INFO(0, y, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_invoke, 0, 0, PhpTensors\\HostArray, 0)
    { "slices", ZEND_TYPE_INIT_MASK(MAY_BE_LONG | MAY_BE_ARRAY | MAY_BE_NULL | _ZEND_ARG_INFO_FLAGS(0, 1, 0)), NULL },
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_slice, 0, 0, PhpTensors\\HostArray, 0)
    { "selectors", ZEND_TYPE_INIT_MASK(MAY_BE_LONG | MAY_BE_STRING | MAY_BE_ARRAY | MAY_BE_NULL | _ZEND_ARG_INFO_FLAGS(0, 1, 0)), NULL },
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_concat, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, tensors, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_binary, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_clamp, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, min, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_NULL, "null")
    ZEND_ARG_TYPE_MASK(0, max, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_NULL, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_clip, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, minimum, MAY_BE_LONG | MAY_BE_DOUBLE, NULL)
    ZEND_ARG_TYPE_MASK(0, maximum, MAY_BE_LONG | MAY_BE_DOUBLE, NULL)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_matmul, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_INFO(0, other, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_transpose, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_ARRAY, 1, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_swapaxes, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, axis1, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, axis2, IS_LONG, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_moveaxis, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, source, MAY_BE_LONG | MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, destination, MAY_BE_LONG | MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_diagonal, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, offset, IS_LONG, 0, "0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis1, IS_LONG, 0, "0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis2, IS_LONG, 0, "1")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_repeat, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, repeats, MAY_BE_LONG | MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_tile, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, reps, MAY_BE_LONG | MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_stack, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, tensors, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_flip, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_ARRAY | MAY_BE_NULL, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_roll, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, shift, MAY_BE_LONG | MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_ARRAY | MAY_BE_NULL, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_power, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_LONG | MAY_BE_DOUBLE, NULL)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_leaky, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, alpha, IS_DOUBLE, 0, "0.01")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_compare, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_DOUBLE, NULL)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_axis, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_stat, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, correction, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_softmax, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 0, "-1")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_layer_norm, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, axes, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, epsilon, IS_DOUBLE, 0, "0.00001")
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, scale, PhpTensors\\HostArray, 1, "null")
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, bias, PhpTensors\\HostArray, 1, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_gather, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_INFO(0, indices, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_scatter, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_INFO(0, indices, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_INFO(0, updates, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_reshape, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_squeeze, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_unsqueeze, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, axis, IS_LONG, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_broadcast, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_full_like, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_INFO(0, prototype, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, value, MAY_BE_LONG | MAY_BE_DOUBLE, NULL)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_eye, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, rows, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, columns, IS_LONG, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, k, IS_LONG, 0, "0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_identity, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, size, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_arange, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, start, MAY_BE_LONG | MAY_BE_DOUBLE, NULL)
    ZEND_ARG_TYPE_MASK(0, stop, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_NULL, "null")
    ZEND_ARG_TYPE_MASK(0, step, MAY_BE_LONG | MAY_BE_DOUBLE, "1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_linspace, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, start, MAY_BE_LONG | MAY_BE_DOUBLE, NULL)
    ZEND_ARG_TYPE_MASK(0, stop, MAY_BE_LONG | MAY_BE_DOUBLE, NULL)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, num, IS_LONG, 0, "50")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, endpoint, _IS_BOOL, 0, "true")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float64\"")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_logspace, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, start, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, stop, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, num, IS_LONG, 0, "50")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, endpoint, _IS_BOOL, 0, "true")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, base, IS_DOUBLE, 0, "10.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_random_uniform, 0, 4, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, minimum, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, maximum, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, seed, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_rand, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_MASK(0, min, MAY_BE_LONG | MAY_BE_DOUBLE, "0")
    ZEND_ARG_TYPE_MASK(0, max, MAY_BE_LONG | MAY_BE_DOUBLE, "1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_randn, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_astype, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_requires_grad, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, requiresGrad, _IS_BOOL, 0, "true")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_host_unsupported_backward, 0, 0, IS_VOID, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, gradient, PhpTensors\\HostArray, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, retainGraph, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_grad, 0, 0, PhpTensors\\HostArray, 1)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_host_unsupported_zero_grad, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_unsupported_detach, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_host_unsupported_item, 0, 0, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL)
ZEND_END_ARG_INFO()

ZEND_FUNCTION(php_tensors_host_array_unsupported);

#if PHP_VERSION_ID >= 80500
#define HOST_ARRAY_UNSUPPORTED_METHOD(name, arginfo, flags) \
    ZEND_RAW_FENTRY(name, zif_php_tensors_host_array_unsupported, arginfo, flags, NULL, NULL)
#else
#define HOST_ARRAY_UNSUPPORTED_METHOD(name, arginfo, flags) \
    ZEND_RAW_FENTRY(name, zif_php_tensors_host_array_unsupported, arginfo, flags)
#endif

#define HOST_ARRAY_UNSUPPORTED_METHODS \
    HOST_ARRAY_UNSUPPORTED_METHOD("fromFile", arginfo_host_unsupported_from_file, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("fromNpy", arginfo_host_unsupported_from_npy, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("where", arginfo_host_unsupported_where, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("__invoke", arginfo_host_unsupported_invoke, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("slice", arginfo_host_unsupported_slice, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("concat", arginfo_host_unsupported_concat, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("maximum", arginfo_host_unsupported_binary, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("minimum", arginfo_host_unsupported_binary, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("clamp", arginfo_host_unsupported_clamp, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("clip", arginfo_host_unsupported_clip, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("matmul", arginfo_host_unsupported_matmul, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("transpose", arginfo_host_unsupported_transpose, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("copy", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("swapaxes", arginfo_host_unsupported_swapaxes, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("moveaxis", arginfo_host_unsupported_moveaxis, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("diagonal", arginfo_host_unsupported_diagonal, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("repeat", arginfo_host_unsupported_repeat, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("tile", arginfo_host_unsupported_tile, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("stack", arginfo_host_unsupported_stack, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("flip", arginfo_host_unsupported_flip, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("roll", arginfo_host_unsupported_roll, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("power", arginfo_host_unsupported_power, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("sqrt", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("exp", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("sigmoid", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("sigmoidDerivative", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("softmaxDerivative", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("relu", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("reluDerivative", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("leakyRelu", arginfo_host_unsupported_leaky, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("leakyReluDerivative", arginfo_host_unsupported_leaky, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("expm1", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("exp2", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("log", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("log1p", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("log2", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("log10", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("sin", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("cos", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("tan", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("asin", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("acos", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("atan", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("sinh", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("cosh", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("tanh", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("tanhDerivative", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("abs", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("neg", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("floor", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("ceil", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("round", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("gt", arginfo_host_unsupported_compare, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("lt", arginfo_host_unsupported_compare, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("eq", arginfo_host_unsupported_compare, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("ne", arginfo_host_unsupported_compare, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("ge", arginfo_host_unsupported_compare, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("le", arginfo_host_unsupported_compare, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("sum", arginfo_host_unsupported_axis, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("mean", arginfo_host_unsupported_axis, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("max", arginfo_host_unsupported_axis, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("min", arginfo_host_unsupported_axis, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("prod", arginfo_host_unsupported_axis, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("cumsum", arginfo_host_unsupported_axis, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("cumprod", arginfo_host_unsupported_axis, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("argMax", arginfo_host_unsupported_axis, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("argMin", arginfo_host_unsupported_axis, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("all", arginfo_host_unsupported_axis, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("any", arginfo_host_unsupported_axis, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("var", arginfo_host_unsupported_stat, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("std", arginfo_host_unsupported_stat, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("softmax", arginfo_host_unsupported_softmax, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("layerNorm", arginfo_host_unsupported_layer_norm, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("gather", arginfo_host_unsupported_gather, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("scatterAdd", arginfo_host_unsupported_scatter, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("reshape", arginfo_host_unsupported_reshape, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("squeeze", arginfo_host_unsupported_squeeze, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("unsqueeze", arginfo_host_unsupported_unsqueeze, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("broadcastTo", arginfo_host_unsupported_broadcast, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("flatten", arginfo_host_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("fullLike", arginfo_host_unsupported_full_like, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("eye", arginfo_host_unsupported_eye, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("identity", arginfo_host_unsupported_identity, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("arange", arginfo_host_unsupported_arange, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("linspace", arginfo_host_unsupported_linspace, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("logspace", arginfo_host_unsupported_logspace, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("randomUniform", arginfo_host_unsupported_random_uniform, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("rand", arginfo_host_unsupported_rand, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("randn", arginfo_host_unsupported_randn, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("astype", arginfo_host_unsupported_astype, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("requiresGrad", arginfo_host_unsupported_requires_grad, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("backward", arginfo_host_unsupported_backward, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("grad", arginfo_host_unsupported_grad, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("zeroGrad", arginfo_host_unsupported_zero_grad, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("detach", arginfo_host_unsupported_detach, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("item", arginfo_host_unsupported_item, ZEND_ACC_PUBLIC)

#endif
