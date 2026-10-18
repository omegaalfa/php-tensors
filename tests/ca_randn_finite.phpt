--TEST--
randn validates finite results for all supported floating dtypes
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

foreach (['float16', 'bfloat16', 'float32', 'float64'] as $dtype) {
    foreach ([1, 65, 128] as $size) {
        $a = CudaArray::randn([$size], $dtype, 0);
        $b = CudaArray::randn([$size], $dtype, 0);
        if ($a->toBuffer() !== $b->toBuffer()) {
            throw new RuntimeException("seed reproducibility failed for $dtype/$size");
        }
        foreach ($a->toArray() as $value) {
            if (!is_finite($value)) {
                throw new RuntimeException("non-finite randn result for $dtype/$size");
            }
        }
    }
    if (CudaArray::randn([0], $dtype, 0)->getSize() !== 0) {
        throw new RuntimeException("empty randn result failed for $dtype");
    }
}

echo "randn finite checks passed\n";
?>
--EXPECT--
randn finite checks passed
