--TEST--
PhpTensors\\CudaArray::clip preserves dtype, layouts and clamp semantics
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;

function clip_check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}

$input = new CudaArray([-3.5, -1.0, 0.5, 4.0], 'float64');
$clipped = $input->clip(-1.0, 2.0);
clip_check($clipped->toArray() === [-1.0, -1.0, 0.5, 2.0], 'basic values');
clip_check($clipped->dtype() === 'float64' && $clipped->getShape() === [4], 'metadata');

$integer = (new CudaArray([-100, -2, 7, 100], 'int8'))->clip(-10, 10);
clip_check($integer->toArray() === [-10, -2, 7, 10], 'integer values');
$unsigned = (new CudaArray([0, 10, 255], 'uint8'))->clip(5, 200);
clip_check($unsigned->toArray() === [5, 10, 200] && $unsigned->dtype() === 'uint8', 'unsigned values');

$view = (new CudaArray([[1, 9, 3], [8, 2, 7]], 'int32'))->slice(':, ::2');
$viewResult = $view->clip(2, 6);
clip_check($viewResult->toArray() === [[2, 3], [6, 6]], 'strided view');

$special = new CudaArray([NAN, -INF, INF], 'float32')->clip(-1.0, 1.0)->toArray();
clip_check(is_nan($special[0]), 'NaN preservation');
clip_check($special[1] === -1.0 && $special[2] === 1.0, 'infinity bounds');

$empty = CudaArray::zeros([0, 3], 'float32')->clip(-1.0, 1.0);
clip_check($empty->getShape() === [0, 3] && $empty->getSize() === 0, 'empty tensor');

$independent = $input->clip(-1.0, 2.0);
$input[0] = 99.0;
clip_check($independent->toArray()[0] === -1.0, 'independent result');

$clamp = (new CudaArray([-2.0, 0.5, 3.0]))->clamp(-1.0, 1.0);
clip_check($clamp->toArray() === [-1.0, 0.5, 1.0], 'clamp preserved');

foreach ([[-2.0, -3.0], [NAN, 1.0], [1, 0.5]] as [$minimum, $maximum]) {
    try {
        (new CudaArray([1.0], 'float32'))->clip($minimum, $maximum);
        throw new RuntimeException('invalid bounds accepted');
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
}

try {
    (new CudaArray([1], 'int32'))->clip(0.5, 1.5);
    throw new RuntimeException('fractional integer bounds accepted');
} catch (PhpTensors\InvalidArgumentException $error) {
}

try {
    (new CudaArray([1], 'float16'))->clip(-1.0, 1.0);
    throw new RuntimeException('float16 clip accepted');
} catch (PhpTensors\InvalidArgumentException $error) {
}

$tracked = (new CudaArray([1.0, 2.0]))->requiresGrad();
try {
    $tracked->clip(0.0, 1.0);
    throw new RuntimeException('Autograd clip accepted');
} catch (PhpTensors\InvalidArgumentException $error) {
}

try {
    Fusion::run(fn() => (new CudaArray([1.0]))->clip(0.0, 1.0));
    throw new RuntimeException('Fusion clip accepted');
} catch (PhpTensors\InvalidArgumentException $error) {
}

try {
    (new PhpTensors\HostArray([1.0]))->clip(0, 1);
    throw new RuntimeException('HostArray clip accepted');
} catch (PhpTensors\RuntimeException $error) {
}

echo "clip checks passed\n";
?>
--EXPECT--
clip checks passed
