--TEST--
PhpTensors CudaArray relu preserves signed-zero and special-value semantics
--SKIPIF--
<?php
if (!extension_loaded('cuda') && !extension_loaded('phptensors')) die('skip CUDA extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

function relu_check(bool $condition, string $message): void {
    if (!$condition) throw new RuntimeException($message);
}
function relu_bits(float $value): string {
    return bin2hex(pack('d', $value));
}
function relu_close(float $actual, float $expected, float $tolerance): bool {
    return abs($actual - $expected) <= $tolerance;
}

$input = CudaArray::fromFlatArray([-2.0, -0.0, 0.0, 2.0, -INF, INF, NAN], [7], 'float64');
$output = $input->relu();
$values = $output->toArray();
relu_check($output !== $input, 'independent object');
relu_check($output->getShape() === [7] && $output->dtype() === 'float64', 'metadata');
relu_check($values[0] === 0.0 && $values[1] === 0.0 && $values[2] === 0.0, 'negative values and zero');
relu_check(relu_bits($values[1]) === relu_bits(0.0) && relu_bits($values[2]) === relu_bits(0.0), 'positive zero output');
relu_check($values[3] === 2.0 && $values[4] === 0.0 && $values[5] === INF, 'infinities');
relu_check($values[6] === 0.0, 'NaN maps to positive zero');

$matrix = new CudaArray([[-1.0, 0.5, -2.0], [3.0, -4.0, 5.0]], 'float32');
$view = $matrix->slice(':, ::2')->relu();
$view_values = $view->toArray();
relu_check($view->getShape() === [2, 2], 'view shape');
relu_check(relu_close($view_values[0][0], 0.0, 1.0e-6) &&
           relu_close($view_values[0][1], 0.0, 1.0e-6) &&
           relu_close($view_values[1][0], 3.0, 1.0e-6) &&
           relu_close($view_values[1][1], 5.0, 1.0e-6), 'strided values');
unset($matrix);
relu_check(count($view->toArray()) === 2, 'view lifetime');

$empty = CudaArray::zeros([0, 3], 'float32')->relu();
relu_check($empty->getShape() === [0, 3] && $empty->getSize() === 0, 'empty result');

try {
    CudaArray::ones([2], 'int32')->relu();
    throw new RuntimeException('integer dtype accepted');
} catch (PhpTensors\InvalidArgumentException $e) {}

try {
    (new CudaArray([1.0], 'float32'))->requiresGrad()->relu();
    throw new RuntimeException('Autograd accepted');
} catch (PhpTensors\InvalidArgumentException $e) {}

try {
    PhpTensors\Fusion::run(fn() => CudaArray::ones([1], 'float32')->relu());
    throw new RuntimeException('Fusion accepted');
} catch (PhpTensors\RuntimeException $e) {}

try {
    (new PhpTensors\HostArray([1.0]))->relu();
    throw new RuntimeException('HostArray execution accepted');
} catch (PhpTensors\RuntimeException $e) {}

echo "relu checked\n";
?>
--EXPECT--
relu checked
