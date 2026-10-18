--TEST--
PhpTensors CudaArray leakyReluDerivative follows original-input semantics
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

function check_leaky_relu_derivative(bool $ok, string $label): void {
    if (!$ok) throw new RuntimeException($label);
}

$input = CudaArray::fromFlatArray([-2.0, 2.0, 0.0, -0.0, NAN, INF, -INF], [7], 'float64');
$default = $input->leakyReluDerivative();
$values = $default->toArray();
check_leaky_relu_derivative($default !== $input, 'independent object');
check_leaky_relu_derivative($default->getShape() === [7] && $default->dtype() === 'float64', 'metadata');
check_leaky_relu_derivative($values === [0.01, 1.0, 0.01, 0.01, 0.01, 1.0, 0.01], 'default semantics');
unset($input);
check_leaky_relu_derivative($default->toArray() === $values, 'independent lifetime');

$zeroAlpha = CudaArray::fromFlatArray([-1.0, 0.0, -0.0, NAN, 1.0], [5], 'float64')->leakyReluDerivative(0.0);
check_leaky_relu_derivative($zeroAlpha->toArray() === [0.0, 0.0, 0.0, 0.0, 1.0], 'zero alpha');

$negativeAlpha = CudaArray::fromFlatArray([-1.0, 0.0, -0.0, NAN, INF, -INF, 1.0], [7], 'float64')->leakyReluDerivative(-0.25);
check_leaky_relu_derivative($negativeAlpha->toArray() === [-0.25, -0.25, -0.25, -0.25, 1.0, -0.25, 1.0], 'negative alpha');

$single = CudaArray::fromFlatArray([-2.0, 2.0, NAN, -0.0], [4], 'float32')->leakyReluDerivative(0.2);
$singleValues = $single->toArray();
check_leaky_relu_derivative($single->dtype() === 'float32' && abs($singleValues[0] - 0.2) < 1e-6 && $singleValues[1] === 1.0 && abs($singleValues[2] - 0.2) < 1e-6 && abs($singleValues[3] - 0.2) < 1e-6, 'float32');

$matrix = new CudaArray([[-2.0, 3.0, -4.0], [5.0, -6.0, 7.0]], 'float32');
$view = $matrix->slice(':, ::2')->leakyReluDerivative(0.5);
check_leaky_relu_derivative($view->getShape() === [2, 2] && $view->toArray() === [[0.5, 0.5], [1.0, 1.0]], 'strided view');
unset($matrix);
check_leaky_relu_derivative($view->toArray() === [[0.5, 0.5], [1.0, 1.0]], 'view result lifetime');

$empty = CudaArray::zeros([0, 3], 'float32')->leakyReluDerivative();
check_leaky_relu_derivative($empty->getShape() === [0, 3] && $empty->getSize() === 0 && $empty->toArray() === [], 'empty tensor');

foreach ([NAN, INF, -INF] as $invalidAlpha) {
    try {
        CudaArray::ones([1], 'float64')->leakyReluDerivative($invalidAlpha);
        throw new RuntimeException('non-finite alpha accepted');
    } catch (PhpTensors\InvalidArgumentException $e) {
    }
}
try {
    CudaArray::ones([1], 'float32')->leakyReluDerivative(1.0e100);
    throw new RuntimeException('out-of-range float32 alpha accepted');
} catch (PhpTensors\InvalidArgumentException $e) {
}
try {
    CudaArray::ones([1], 'int32')->leakyReluDerivative();
    throw new RuntimeException('integer dtype accepted');
} catch (PhpTensors\InvalidArgumentException $e) {
}

try {
    PhpTensors\Fusion::run(fn() => $default->leakyReluDerivative());
    throw new RuntimeException('Fusion accepted leakyReluDerivative');
} catch (PhpTensors\RuntimeException $e) {
}

$tracked = CudaArray::fromFlatArray([1.0], [1], 'float32')->requiresGrad();
try {
    $tracked->leakyReluDerivative();
    throw new RuntimeException('Autograd accepted leakyReluDerivative');
} catch (PhpTensors\InvalidArgumentException $e) {
}

echo "leakyReluDerivative checked\n";
?>
--EXPECT--
leakyReluDerivative checked
