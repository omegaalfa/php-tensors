--TEST--
PhpTensors CudaArray softmaxDerivative computes the diagonal Jacobian approximation
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

function check_softmax_derivative(bool $ok, string $label): void {
    if (!$ok) throw new RuntimeException($label);
}

$input = CudaArray::fromFlatArray([0.0, 0.25, 0.5, 0.75, 1.0], [5], 'float64');
$result = $input->softmaxDerivative();
$values = $result->toArray();
check_softmax_derivative($result !== $input, 'independent object');
check_softmax_derivative($result->getShape() === [5] && $result->dtype() === 'float64', 'metadata');
check_softmax_derivative($values[0] === 0.0 && abs($values[1] - 0.1875) < 1e-12 && $values[2] === 0.25 && abs($values[3] - 0.1875) < 1e-12 && $values[4] === 0.0, 'known values');
unset($input);
check_softmax_derivative($result->toArray() === $values, 'independent lifetime');

$outside = CudaArray::fromFlatArray([-1.0, 2.0, NAN, INF, -INF], [5], 'float64')->softmaxDerivative()->toArray();
check_softmax_derivative($outside[0] === -2.0 && $outside[1] === -2.0, 'no implicit clamp');
check_softmax_derivative(is_nan($outside[2]), 'NaN');
check_softmax_derivative(is_infinite($outside[3]) && $outside[3] < 0, 'positive infinity');
check_softmax_derivative(is_infinite($outside[4]) && $outside[4] < 0, 'negative infinity');

$single = CudaArray::fromFlatArray([0.25, 0.5, 0.75], [3], 'float32')->softmaxDerivative();
$singleValues = $single->toArray();
check_softmax_derivative($single->dtype() === 'float32' && abs($singleValues[0] - 0.1875) < 1e-6 && abs($singleValues[1] - 0.25) < 1e-6 && abs($singleValues[2] - 0.1875) < 1e-6, 'float32');

$matrix = new CudaArray([[0.25, 0.5, 0.75], [1.0, 0.0, 0.5]], 'float32');
$view = $matrix->slice(':, ::2')->softmaxDerivative();
$viewValues = $view->toArray();
check_softmax_derivative($view->getShape() === [2, 2] && abs($viewValues[0][0] - 0.1875) < 1e-6 && abs($viewValues[0][1] - 0.1875) < 1e-6, 'strided view');
unset($matrix);
check_softmax_derivative(count($view->toArray()) === 2, 'view lifetime');

$empty = CudaArray::zeros([0, 3], 'float32')->softmaxDerivative();
check_softmax_derivative($empty->getShape() === [0, 3] && $empty->getSize() === 0 && $empty->toArray() === [], 'empty tensor');

try {
    CudaArray::ones([2], 'int32')->softmaxDerivative();
    throw new RuntimeException('integer dtype accepted');
} catch (PhpTensors\InvalidArgumentException $e) {
}
try {
    (new PhpTensors\HostArray([1.0]))->softmaxDerivative();
    throw new RuntimeException('HostArray accepted softmaxDerivative');
} catch (PhpTensors\RuntimeException $e) {
}
try {
    PhpTensors\Fusion::run(fn() => $result->softmaxDerivative());
    throw new RuntimeException('Fusion accepted softmaxDerivative');
} catch (PhpTensors\RuntimeException $e) {
}

$tracked = CudaArray::fromFlatArray([0.5], [1], 'float32')->requiresGrad();
try {
    $tracked->softmaxDerivative();
    throw new RuntimeException('Autograd accepted softmaxDerivative');
} catch (PhpTensors\InvalidArgumentException $e) {
}

echo "softmaxDerivative checked\n";
?>
--EXPECT--
softmaxDerivative checked
