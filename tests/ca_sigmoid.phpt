--TEST--
PhpTensors CudaArray sigmoid is stable, strided and float-only
--SKIPIF--
<?php
if (!extension_loaded('cuda') && !extension_loaded('phptensors')) die('skip CUDA extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

function sigmoid_check(bool $condition, string $message): void {
    if (!$condition) throw new RuntimeException($message);
}
function sigmoid_close(float $actual, float $expected, float $tolerance): bool {
    return abs($actual - $expected) <= $tolerance;
}

$input = CudaArray::fromFlatArray([-1000.0, -1.0, 0.0, 1.0, 1000.0], [5], 'float64');
$output = $input->sigmoid();
$values = $output->toArray();
sigmoid_check($output !== $input, 'independent object');
sigmoid_check($output->getShape() === [5] && $output->dtype() === 'float64', 'metadata');
sigmoid_check($values[0] === 0.0 && $values[4] === 1.0, 'extreme values');
sigmoid_check(sigmoid_close($values[1], 0.2689414213699951, 1.0e-12), 'negative value');
sigmoid_check($values[2] === 0.5, 'zero');
sigmoid_check(sigmoid_close($values[3], 0.7310585786300049, 1.0e-12), 'positive value');

$special = CudaArray::fromFlatArray([NAN, -INF, INF], [3], 'float64')->sigmoid()->toArray();
sigmoid_check(is_nan($special[0]), 'NaN propagation');
sigmoid_check($special[1] === 0.0 && $special[2] === 1.0, 'infinity handling');

$matrix = new CudaArray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], 'float32');
$view = $matrix->slice(':, ::2')->sigmoid();
$view_values = $view->toArray();
sigmoid_check($view->getShape() === [2, 2], 'view shape');
sigmoid_check(sigmoid_close($view_values[0][0], 0.5, 1.0e-6) &&
              sigmoid_close($view_values[1][1], 0.993307149, 1.0e-6), 'strided values');

$empty = CudaArray::zeros([0, 3], 'float32')->sigmoid();
sigmoid_check($empty->getShape() === [0, 3] && $empty->getSize() === 0, 'empty result');

try {
    CudaArray::ones([2], 'int32')->sigmoid();
    throw new RuntimeException('integer dtype accepted');
} catch (PhpTensors\InvalidArgumentException $e) {}

try {
    (new CudaArray([1.0], 'float32'))->requiresGrad()->sigmoid();
    throw new RuntimeException('Autograd accepted');
} catch (PhpTensors\InvalidArgumentException $e) {}

try {
    PhpTensors\Fusion::run(fn() => CudaArray::ones([1], 'float32')->sigmoid());
    throw new RuntimeException('Fusion accepted');
} catch (PhpTensors\RuntimeException $e) {}

try {
    (new PhpTensors\HostArray([1.0]))->sigmoid();
    throw new RuntimeException('HostArray execution accepted');
} catch (PhpTensors\RuntimeException $e) {}

echo "sigmoid checked\n";
?>
--EXPECT--
sigmoid checked
