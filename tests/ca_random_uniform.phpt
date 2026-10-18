--TEST--
PhpTensors\\CudaArray randomUniform is reproducible and respects its interval
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

$a = CudaArray::randomUniform([2, 3, 4], -2.5, 7.5, 12345, 'float32');
$b = CudaArray::randomUniform([2, 3, 4], -2.5, 7.5, 12345, 'float32');
$c = CudaArray::randomUniform([2, 3, 4], -2.5, 7.5, 54321, 'float32');
if ($a->getShape() !== [2, 3, 4] || $a->getSize() !== 24 ||
    $a->toArray() !== $b->toArray() || $a->toArray() === $c->toArray()) {
    echo "seed or shape mismatch\n";
    exit(1);
}
foreach ($a->toArray() as $row) foreach ($row as $column) foreach ($column as $value) {
    if (!is_finite($value) || $value < -2.5 || $value >= 7.5) {
        echo "float32 interval mismatch\n";
        exit(1);
    }
}

$double = CudaArray::randomUniform([1000], 0.25, 0.50, 999, 'float64')->toArray();
$sum = 0.0;
foreach ($double as $value) {
    if (!is_finite($value) || $value < 0.25 || $value >= 0.50) {
        echo "float64 interval mismatch\n";
        exit(1);
    }
    $sum += $value;
}
if (abs($sum / count($double) - 0.375) > 0.03) {
    echo "distribution mismatch\n";
    exit(1);
}

if (CudaArray::randomUniform([0, 4], -1.0, 1.0, 7, 'float64')->getSize() !== 0) {
    echo "empty mismatch\n";
    exit(1);
}
$recreatedA = CudaArray::randomUniform([32], -1.0, 1.0, 2024, 'float64')->toArray();
unset($a, $b, $c);
$recreatedB = CudaArray::randomUniform([32], -1.0, 1.0, 2024, 'float64')->toArray();
if ($recreatedA !== $recreatedB) {
    echo "recreation mismatch\n";
    exit(1);
}

$narrowFloat32 = CudaArray::randomUniform([32], 1.0, 1.0000001192092896, 17, 'float32')->toArray();
$narrowFloat64 = CudaArray::randomUniform([32], 1.0, 1.0000000000000002, 17, 'float64')->toArray();
foreach ($narrowFloat32 as $value) if ($value < 1.0 || $value >= 1.0000001192092896) { echo "narrow float32 mismatch\n"; exit(1); }
foreach ($narrowFloat64 as $value) if ($value < 1.0 || $value >= 1.0000000000000002) { echo "narrow float64 mismatch\n"; exit(1); }

echo "randomUniform checks passed\n";
?>
--EXPECT--
randomUniform checks passed
