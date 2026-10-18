--TEST--
CudaArray::grad() returns an independent gradient snapshot
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

function check_grad_copy(bool $condition, string $message): void
{
    if (!$condition) {
        throw new RuntimeException($message);
    }
}

$before = CudaArray::fromFlatArray([1.0], [1])->requiresGrad();
check_grad_copy($before->grad() === null, 'grad() before backward must be null');

$parameter = CudaArray::fromFlatArray([2.0, 3.0], [2])->requiresGrad();
$loss = $parameter->sum();
$loss->backward();

$first = $parameter->grad();
$second = $parameter->grad();
check_grad_copy($first !== null && $second !== null, 'gradient was not returned');
check_grad_copy($first !== $second, 'grad() returned the same PHP object');
check_grad_copy($first->toArray() === [1.0, 1.0], 'unexpected gradient');

$first[0] = 9.0;
check_grad_copy($parameter->grad()->toArray() === [1.0, 1.0], 'gradient copy aliased internal state');
check_grad_copy($second->toArray() === [1.0, 1.0], 'gradient snapshots share storage');

$parameter->zeroGrad();
check_grad_copy($parameter->grad() === null, 'zeroGrad() did not clear internal gradient');
check_grad_copy($first->toArray() === [9.0, 1.0], 'first snapshot was invalidated');
check_grad_copy($second->toArray() === [1.0, 1.0], 'second snapshot was invalidated');

$lossAgain = $parameter->sum();
$lossAgain->backward();
check_grad_copy($parameter->grad()->toArray() === [1.0, 1.0], 'backward after zeroGrad() failed');

$left = CudaArray::fromFlatArray([1.0, 2.0], [2, 1])->requiresGrad();
$right = CudaArray::fromFlatArray([3.0, 4.0, 5.0], [1, 3])->requiresGrad();
($left + $right)->sum()->backward();
check_grad_copy($left->grad()->toArray() === [[3.0], [3.0]], 'broadcast left gradient mismatch');
check_grad_copy($right->grad()->toArray() === [[2.0, 2.0, 2.0]], 'broadcast right gradient mismatch');

$viewBase = CudaArray::fromFlatArray([1.0, 2.0, 3.0, 4.0], [2, 2])->requiresGrad();
$view = $viewBase->slice(':, 0:1');
$view->sum()->backward();
$viewSnapshot = $viewBase->grad();
check_grad_copy($viewSnapshot->toArray() === [[1.0, 0.0], [1.0, 0.0]], 'view gradient mismatch');

foreach (['float16', 'bfloat16'] as $dtype) {
    try {
        $low = CudaArray::fromFlatArray([2.0], [1], $dtype)->requiresGrad();
        ($low * $low)->sum()->backward();
        check_grad_copy($low->grad()->dtype() === 'float32', $dtype . ' gradient must be float32');
    } catch (Throwable $e) {
        if (!$e instanceof PhpTensors\InvalidArgumentException) {
            throw $e;
        }
    }
}

unset($parameter, $loss, $lossAgain, $before);
check_grad_copy($first->toArray() === [9.0, 1.0], 'snapshot failed after parameter destruction');
check_grad_copy($second->toArray() === [1.0, 1.0], 'snapshot failed after parameter destruction');

echo "ok\n";
?>
--EXPECT--
ok
