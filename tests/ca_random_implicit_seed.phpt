--TEST--
Implicit random seeds remain distinct while explicit zero remains reproducible
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

$explicitRandA = CudaArray::rand([64], -1.0, 1.0, 'float32', 0);
$explicitRandB = CudaArray::rand([64], -1.0, 1.0, 'float32', 0);
if ($explicitRandA->toBuffer() !== $explicitRandB->toBuffer()) {
    throw new RuntimeException('explicit rand seed zero is not reproducible');
}

$explicitRandnA = CudaArray::randn([65], 'float64', 0);
$explicitRandnB = CudaArray::randn([65], 'float64', 0);
if ($explicitRandnA->toBuffer() !== $explicitRandnB->toBuffer()) {
    throw new RuntimeException('explicit randn seed zero is not reproducible');
}

$implicitRandA = CudaArray::rand([64]);
$implicitRandB = CudaArray::rand([64]);
$implicitRandnA = CudaArray::randn([65]);
$implicitRandnB = CudaArray::randn([65]);
if ($implicitRandA->toBuffer() === $implicitRandB->toBuffer() ||
    $implicitRandnA->toBuffer() === $implicitRandnB->toBuffer()) {
    throw new RuntimeException('implicit random seed was reused');
}

unset($implicitRandA, $implicitRandB, $implicitRandnA, $implicitRandnB);
$afterReuseA = CudaArray::randn([65]);
$afterReuseB = CudaArray::randn([65]);
if ($afterReuseA->toBuffer() === $afterReuseB->toBuffer()) {
    throw new RuntimeException('implicit seed repeated after tensor destruction');
}

echo "implicit random seed checks passed\n";
?>
--EXPECT--
implicit random seed checks passed
