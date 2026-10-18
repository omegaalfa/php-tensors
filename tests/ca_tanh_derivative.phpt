--TEST--
PhpTensors CudaArray tanhDerivative uses activated-value semantics
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

function check_tanh_derivative(bool $ok, string $label): void {
    if (!$ok) throw new RuntimeException($label);
}

$input = CudaArray::fromFlatArray([-1.0, -0.5, -0.0, 0.0, 0.5, 1.0], [6], 'float64');
$result = $input->tanhDerivative();
$values = $result->toArray();
check_tanh_derivative($result !== $input, 'independent object');
check_tanh_derivative($result->getShape() === [6] && $result->dtype() === 'float64', 'metadata');
check_tanh_derivative($values === [0.0, 0.75, 1.0, 1.0, 0.75, 0.0], 'known values');
unset($input);
check_tanh_derivative($result->toArray() === $values, 'independent lifetime');

$outside = CudaArray::fromFlatArray([-2.0, 2.0], [2], 'float64')->tanhDerivative()->toArray();
check_tanh_derivative($outside === [-3.0, -3.0], 'no implicit clamp');

$special = CudaArray::fromFlatArray([NAN, -INF, INF], [3], 'float64')->tanhDerivative()->toArray();
check_tanh_derivative(is_nan($special[0]), 'NaN');
check_tanh_derivative(is_infinite($special[1]) && $special[1] < 0, 'negative infinity');
check_tanh_derivative(is_infinite($special[2]) && $special[2] < 0, 'positive infinity');

$single = CudaArray::fromFlatArray([-0.5, 0.0, 0.5], [3], 'float32')->tanhDerivative();
$singleValues = $single->toArray();
check_tanh_derivative($single->dtype() === 'float32' && abs($singleValues[0] - 0.75) < 1e-6 && $singleValues[1] === 1.0 && abs($singleValues[2] - 0.75) < 1e-6, 'float32');

$matrix = new CudaArray([[-0.5, 0.0, 0.5], [1.0, -1.0, 0.25]], 'float32');
$view = $matrix->slice(':, ::2')->tanhDerivative();
check_tanh_derivative($view->getShape() === [2, 2] && abs($view->toArray()[0][0] - 0.75) < 1e-6 && abs($view->toArray()[0][1] - 0.75) < 1e-6, 'strided view');
unset($matrix);
check_tanh_derivative(count($view->toArray()) === 2, 'view lifetime');

$empty = CudaArray::zeros([0, 3], 'float32')->tanhDerivative();
check_tanh_derivative($empty->getShape() === [0, 3] && $empty->getSize() === 0 && $empty->toArray() === [], 'empty tensor');

try {
    CudaArray::ones([2], 'int32')->tanhDerivative();
    throw new RuntimeException('integer dtype accepted');
} catch (PhpTensors\InvalidArgumentException $e) {
}
try {
    (new PhpTensors\HostArray([1.0]))->tanhDerivative();
    throw new RuntimeException('HostArray accepted tanhDerivative');
} catch (PhpTensors\RuntimeException $e) {
}

try {
    PhpTensors\Fusion::run(fn() => $result->tanhDerivative());
    throw new RuntimeException('Fusion accepted tanhDerivative');
} catch (PhpTensors\RuntimeException $e) {
}

$tracked = CudaArray::fromFlatArray([1.0], [1], 'float32')->requiresGrad();
try {
    $tracked->tanhDerivative();
    throw new RuntimeException('Autograd accepted tanhDerivative');
} catch (PhpTensors\InvalidArgumentException $e) {
}

echo "tanhDerivative checked\n";
?>
--EXPECT--
tanhDerivative checked
