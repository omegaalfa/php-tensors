--TEST--
PhpTensors CudaArray sigmoidDerivative uses activated values and preserves layouts
--SKIPIF--
<?php
if (!extension_loaded('cuda') && !extension_loaded('phptensors')) die('skip CUDA extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

function sigmoid_derivative_check(bool $condition, string $message): void {
    if (!$condition) throw new RuntimeException($message);
}
function sigmoid_derivative_close(float $actual, float $expected, float $tolerance): bool {
    return abs($actual - $expected) <= $tolerance;
}

$activated = CudaArray::fromFlatArray([0.0, 0.25, 0.5, 0.75, 1.0], [5], 'float64');
$derivative = $activated->sigmoidDerivative();
$values = $derivative->toArray();
sigmoid_derivative_check($derivative !== $activated, 'independent object');
sigmoid_derivative_check($derivative->getShape() === [5] && $derivative->dtype() === 'float64', 'metadata');
sigmoid_derivative_check($values[0] === 0.0 && $values[4] === 0.0, 'endpoint derivatives');
sigmoid_derivative_check($values[2] === 0.25, 'middle derivative');
sigmoid_derivative_check(sigmoid_derivative_close($values[1], 0.1875, 1.0e-12) &&
                         sigmoid_derivative_close($values[3], 0.1875, 1.0e-12), 'quarter values');

$outside = CudaArray::fromFlatArray([-1.0, 2.0], [2], 'float64')->sigmoidDerivative()->toArray();
sigmoid_derivative_check($outside[0] === -2.0 && $outside[1] === -2.0, 'no implicit clamp');

$special = CudaArray::fromFlatArray([NAN, -INF, INF], [3], 'float64')->sigmoidDerivative()->toArray();
sigmoid_derivative_check(is_nan($special[0]), 'NaN propagation');
sigmoid_derivative_check(is_infinite($special[1]) && $special[1] < 0 &&
                         is_infinite($special[2]) && $special[2] < 0, 'infinity arithmetic');

$matrix = new CudaArray([[0.0, 0.5, 1.0], [0.25, 0.75, 1.0]], 'float32');
$view = $matrix->slice(':, ::2')->sigmoidDerivative();
$view_values = $view->toArray();
sigmoid_derivative_check($view->getShape() === [2, 2], 'view shape');
sigmoid_derivative_check(sigmoid_derivative_close($view_values[0][0], 0.0, 1.0e-6) &&
                         sigmoid_derivative_close($view_values[0][1], 0.0, 1.0e-6) &&
                         sigmoid_derivative_close($view_values[1][0], 0.1875, 1.0e-6), 'strided values');
unset($matrix);
sigmoid_derivative_check(count($view->toArray()) === 2, 'view lifetime');

$empty = CudaArray::zeros([0, 3], 'float32')->sigmoidDerivative();
sigmoid_derivative_check($empty->getShape() === [0, 3] && $empty->getSize() === 0, 'empty result');

try {
    CudaArray::ones([2], 'int32')->sigmoidDerivative();
    throw new RuntimeException('integer dtype accepted');
} catch (PhpTensors\InvalidArgumentException $e) {}

try {
    (new CudaArray([0.5], 'float32'))->requiresGrad()->sigmoidDerivative();
    throw new RuntimeException('Autograd accepted');
} catch (PhpTensors\InvalidArgumentException $e) {}

try {
    PhpTensors\Fusion::run(fn() => CudaArray::ones([1], 'float32')->sigmoidDerivative());
    throw new RuntimeException('Fusion accepted');
} catch (PhpTensors\RuntimeException $e) {}

try {
    (new PhpTensors\HostArray([0.5]))->sigmoidDerivative();
    throw new RuntimeException('HostArray execution accepted');
} catch (PhpTensors\RuntimeException $e) {}

echo "sigmoidDerivative checked\n";
?>
--EXPECT--
sigmoidDerivative checked
