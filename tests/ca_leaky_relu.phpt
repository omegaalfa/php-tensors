--TEST--
PhpTensors CudaArray leakyRelu applies a configurable finite negative slope
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

function check_leaky_relu(bool $ok, string $label): void {
    if (!$ok) throw new RuntimeException($label);
}
function close_leaky_relu(float $actual, float $expected, float $tol = 1.0e-6): bool {
    return abs($actual - $expected) <= $tol;
}
function bits_leaky_relu(float $value): string {
    return bin2hex(pack('d', $value));
}

$input = CudaArray::fromFlatArray([-4.0, -2.0, -0.0, 0.0, 2.0, 4.0], [6], 'float64');
$default = $input->leakyRelu();
$values = $default->toArray();
check_leaky_relu($default !== $input, 'new object');
check_leaky_relu($default->getShape() === [6] && $default->dtype() === 'float64', 'shape and dtype');
check_leaky_relu(close_leaky_relu($values[0], -0.04) && close_leaky_relu($values[1], -0.02), 'default slope');
check_leaky_relu(bits_leaky_relu($values[2]) === bits_leaky_relu(-0.0) && $values[3] === 0.0, 'signed zeros');
check_leaky_relu($values[4] === 2.0 && $values[5] === 4.0, 'positive branch');
unset($input);
check_leaky_relu(close_leaky_relu($default->toArray()[0], -0.04), 'independent lifetime');

$custom = CudaArray::fromFlatArray([-2.0, -0.0, 0.0, 3.0], [4], 'float64')->leakyRelu(0.25)->toArray();
check_leaky_relu($custom === [-0.5, -0.0, 0.0, 3.0], 'custom slope');

$negativeAlpha = CudaArray::fromFlatArray([-2.0, -0.0, 0.0, 2.0], [4], 'float64')->leakyRelu(-0.5)->toArray();
check_leaky_relu($negativeAlpha[0] === 1.0 && bits_leaky_relu($negativeAlpha[1]) === bits_leaky_relu(0.0), 'negative finite alpha');

$special = CudaArray::fromFlatArray([-INF, INF, NAN], [3], 'float64')->leakyRelu()->toArray();
check_leaky_relu(is_infinite($special[0]) && $special[0] < 0, 'negative infinity');
check_leaky_relu(is_infinite($special[1]) && $special[1] > 0, 'positive infinity');
check_leaky_relu(is_nan($special[2]), 'NaN propagation');

$single = CudaArray::fromFlatArray([-3.0, 2.0], [2], 'float32')->leakyRelu(0.1);
check_leaky_relu($single->dtype() === 'float32' && close_leaky_relu($single->toArray()[0], -0.3), 'float32');

$matrix = new CudaArray([[-2.0, 1.0, -4.0], [3.0, -6.0, 2.0]], 'float32');
$view = $matrix->slice(':, ::2')->leakyRelu(0.5);
check_leaky_relu($view->getShape() === [2, 2] && $view->toArray() === [[-1.0, -2.0], [3.0, 2.0]], 'strided view');
unset($matrix);
check_leaky_relu($view->toArray() === [[-1.0, -2.0], [3.0, 2.0]], 'view result lifetime');

$empty = CudaArray::zeros([0, 3], 'float32')->leakyRelu();
check_leaky_relu($empty->getShape() === [0, 3] && $empty->getSize() === 0 && $empty->toArray() === [], 'empty tensor');

foreach ([NAN, INF, -INF] as $invalidAlpha) {
    try {
        CudaArray::ones([1], 'float64')->leakyRelu($invalidAlpha);
        throw new RuntimeException('non-finite alpha accepted');
    } catch (PhpTensors\InvalidArgumentException $e) {
    }
}
try {
    CudaArray::ones([1], 'float32')->leakyRelu(1.0e100);
    throw new RuntimeException('out-of-range float32 alpha accepted');
} catch (PhpTensors\InvalidArgumentException $e) {
}
try {
    CudaArray::ones([1], 'int32')->leakyRelu();
    throw new RuntimeException('integer dtype accepted');
} catch (PhpTensors\InvalidArgumentException $e) {
}

try {
    PhpTensors\Fusion::run(fn() => $default->leakyRelu());
    throw new RuntimeException('Fusion accepted leakyRelu');
} catch (PhpTensors\RuntimeException $e) {
}

$tracked = CudaArray::fromFlatArray([1.0], [1], 'float32')->requiresGrad();
try {
    $tracked->leakyRelu();
    throw new RuntimeException('Autograd accepted leakyRelu');
} catch (PhpTensors\InvalidArgumentException $e) {
}

echo "leakyRelu checked\n";
?>
--EXPECT--
leakyRelu checked
