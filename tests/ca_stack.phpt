--TEST--
PhpTensors\\CudaArray::stack supports axes, views and independent storage
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

function stack_check(bool $condition, string $message): void
{
    if (!$condition) {
        throw new RuntimeException($message);
    }
}

$a = new CudaArray([[1, 2], [3, 4]], 'int32');
$b = new CudaArray([[5, 6], [7, 8]], 'int32');

$axis0 = CudaArray::stack([$a, $b]);
stack_check($axis0->getShape() === [2, 2, 2], 'axis 0 shape');
stack_check($axis0->toArray() === [
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
], 'axis 0 values');

$axis1 = CudaArray::stack([$a, $b], 1);
stack_check($axis1->getShape() === [2, 2, 2], 'axis 1 shape');
stack_check($axis1->toArray() === [
    [[1, 2], [5, 6]],
    [[3, 4], [7, 8]],
], 'axis 1 values');

$axisLast = CudaArray::stack([$a, $b], -1);
stack_check($axisLast->toArray() === [
    [[1, 5], [2, 6]],
    [[3, 7], [4, 8]],
], 'negative axis values');

$viewA = $a->slice(':, ::2');
$viewB = $b->slice(':, ::2');
$views = CudaArray::stack([$viewA, $viewB], -1);
stack_check($views->getShape() === [2, 1, 2], 'view shape');
stack_check($views->toArray() === [[[1, 5]], [[3, 7]]], 'view values');
$a[0][0] = 99;
stack_check($views->toArray() === [[[1, 5]], [[3, 7]]], 'independent storage');

$empty = CudaArray::stack([
    CudaArray::zeros([0, 2], 'float32'),
    CudaArray::zeros([0, 2], 'float32'),
]);
stack_check($empty->getShape() === [2, 0, 2] && $empty->getSize() === 0, 'empty shape');

foreach ([[], [$a, new CudaArray([1, 2], 'int32')],
          [$a, new CudaArray([[1, 2], [3, 4]], 'float32')],
          [$a, [1, 2]]] as $invalid) {
    try {
        CudaArray::stack($invalid);
        throw new RuntimeException('invalid stack accepted');
    } catch (PhpTensors\InvalidArgumentException $e) {
    }
}

foreach ([3, -4] as $invalidAxis) {
    try {
        CudaArray::stack([$a, $b], $invalidAxis);
        throw new RuntimeException('invalid axis accepted');
    } catch (PhpTensors\InvalidArgumentException $e) {
    }
}

try {
    PhpTensors\HostArray::stack([]);
    throw new RuntimeException('HostArray stack accepted');
} catch (PhpTensors\RuntimeException $e) {
}

unset($a, $b, $viewA, $viewB);
stack_check($axis0->toArray()[0][0][0] === 1, 'source lifetime');
echo "stack checks passed\n";
?>
--EXPECT--
stack checks passed
