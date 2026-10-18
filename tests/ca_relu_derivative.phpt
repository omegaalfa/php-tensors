--TEST--
PhpTensors CudaArray reluDerivative follows original-input semantics
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

function check_relu_derivative(bool $ok, string $label): void {
    if (!$ok) throw new RuntimeException($label);
}

$input = CudaArray::fromFlatArray([-2.0, -0.0, 0.0, 2.0, -INF, INF, NAN], [7], 'float64');
$result = $input->reluDerivative();
$values = $result->toArray();
check_relu_derivative($result !== $input, 'independent object');
check_relu_derivative($result->getShape() === [7], 'shape');
check_relu_derivative($result->dtype() === 'float64', 'float64 dtype');
check_relu_derivative($values === [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], 'float64 values');
unset($input);
check_relu_derivative($result->toArray() === [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], 'result lifetime');

$single = CudaArray::fromFlatArray([-3.0, 0.0, 2.0], [3], 'float32')->reluDerivative();
check_relu_derivative($single->dtype() === 'float32', 'float32 dtype');
check_relu_derivative($single->toArray() === [0.0, 0.0, 1.0], 'float32 values');

$matrix = new CudaArray([[-2.0, 1.0, -3.0], [4.0, -5.0, 6.0]], 'float32');
$viewResult = $matrix->slice(':, ::2')->reluDerivative();
check_relu_derivative($viewResult->getShape() === [2, 2], 'view shape');
check_relu_derivative($viewResult->toArray() === [[0.0, 0.0], [1.0, 1.0]], 'view values');
unset($matrix);
check_relu_derivative($viewResult->toArray() === [[0.0, 0.0], [1.0, 1.0]], 'view result lifetime');

$empty = CudaArray::zeros([0, 3], 'float32')->reluDerivative();
check_relu_derivative($empty->getShape() === [0, 3] && $empty->getSize() === 0, 'empty metadata');
check_relu_derivative($empty->toArray() === [], 'empty values');

try {
    CudaArray::fromFlatArray([-1, 2], [2], 'int32')->reluDerivative();
    throw new RuntimeException('integer dtype accepted');
} catch (PhpTensors\InvalidArgumentException $e) {
}

$tracked = CudaArray::fromFlatArray([1.0], [1], 'float32')->requiresGrad();
try {
    $tracked->reluDerivative();
    throw new RuntimeException('Autograd input accepted');
} catch (PhpTensors\InvalidArgumentException $e) {
}

echo "reluDerivative checked\n";
?>
--EXPECT--
reluDerivative checked
