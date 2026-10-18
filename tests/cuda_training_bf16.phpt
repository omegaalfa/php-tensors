--TEST--
CUDA BF16 forward/backward and AdamW compatibility audit
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\InvalidArgumentException;
use PhpTensors\Optimizer;

function bf16Fail(string $message): never
{
    throw new RuntimeException($message);
}

$lowDtype = 'bfloat16';
$xLow = CudaArray::fromFlatArray([1, 2, 3, 4], [4, 1], $lowDtype);
$targetLow = CudaArray::fromFlatArray([2, 4, 6, 8], [4, 1], $lowDtype);
$lowParameter = CudaArray::fromFlatArray([0.5], [1, 1], $lowDtype)->requiresGrad();
$lowLoss = $xLow->matmul($lowParameter)->subtract($targetLow);
$lowLoss = $lowLoss->multiply($lowLoss)->mean();
$lowInitialLoss = (float)$lowLoss->item();
$lowLoss->backward();
$lowGradient = $lowParameter->grad();
if ($lowGradient === null || $lowGradient->dtype() !== 'float32' ||
    $lowGradient->getShape() !== [1, 1] ||
    !is_finite((float)$lowGradient->item())) {
    bf16Fail('low-precision gradient was not finite FP32 with the expected shape');
}

$optimizer = Optimizer::adamW(0.1, 0.9, 0.999, 1e-8, 0.0);
$optimizerRejected = false;
try {
    $optimizer->initState([$lowParameter]);
} catch (InvalidArgumentException) {
    $optimizerRejected = true;
}
if (!$optimizerRejected) {
    bf16Fail('AdamW unexpectedly accepted a low-precision parameter');
}

// Manual master-weight path: cast the low-precision data to FP32 and keep the
// trainable parameter in FP32. This is ordinary explicit FP32 master storage,
// not automatic mixed-precision support.
$x = $xLow->astype('float32');
$target = $targetLow->astype('float32');
$parameter = CudaArray::fromFlatArray([0.5], [1, 1], 'float32')->requiresGrad();
$optimizer = Optimizer::adamW(0.1, 0.9, 0.999, 1e-8, 0.0);
$state = $optimizer->initState([$parameter]);
if ($state['firstMoment'][0]->dtype() !== 'float32' ||
    $state['secondMoment'][0]->dtype() !== 'float32' ||
    $state['beta1Power']->dtype() !== 'float32' ||
    $state['beta2Power']->dtype() !== 'float32') {
    bf16Fail('AdamW FP32 master state dtype mismatch');
}
$initialLoss = null;
$finalLoss = null;
for ($step = 0; $step < 20; $step++) {
    $optimizer->zeroGrad([$parameter]);
    $prediction = $x->matmul($parameter);
    $difference = $prediction->subtract($target);
    $loss = $difference->multiply($difference)->mean();
    $value = (float)$loss->item();
    $initialLoss ??= $value;
    $finalLoss = $value;
    $loss->backward();
    $gradient = $parameter->grad();
    if ($gradient === null || $gradient->dtype() !== 'float32' ||
        !is_finite((float)$gradient->item())) {
        bf16Fail('FP32 master gradient was invalid');
    }
    $result = $optimizer->step([$parameter], [$gradient], $state);
    $parameter = $result['parameters'][0]->requiresGrad();
    $state = $result['state'];
}
if ($finalLoss >= $initialLoss * 0.25) {
    bf16Fail("manual FP32-master loss did not decrease: $initialLoss -> $finalLoss");
}

$limits = CudaArray::fromFlatArray([3.389531389e38, 2 ** -133, 2 ** -134], [3], $lowDtype);
$overflow = $limits->multiply(CudaArray::full([3], 2.0, $lowDtype))->toArray();
$underflow = $limits->multiply(CudaArray::full([3], 0.5, $lowDtype))->toArray();
if (!is_finite((float)$limits->toArray()[0]) ||
    $underflow[2] != 0.0 || is_finite((float)$overflow[0])) {
    bf16Fail('low-precision overflow/underflow behavior was not observable as expected');
}

echo "dtype=$lowDtype\n";
echo "low-precision gradient=float32\n";
echo "AdamW low-precision parameters: rejected\n";
echo "manual FP32 master: loss decreased\n";
echo "steps=20\n";
echo "overflow=nonfinite underflow=zero\n";
?>
--EXPECT--
dtype=bfloat16
low-precision gradient=float32
AdamW low-precision parameters: rejected
manual FP32 master: loss decreased
steps=20
overflow=nonfinite underflow=zero
