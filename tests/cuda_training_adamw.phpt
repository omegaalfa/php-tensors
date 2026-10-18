--TEST--
CUDA integrated AdamW training keeps finite gradients and lowers loss
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\Optimizer;

function failTraining(string $message): never
{
    throw new RuntimeException($message);
}

$x = CudaArray::fromFlatArray([1, 2, 3, 4], [4, 1], 'float32');
$target = CudaArray::fromFlatArray([2, 4, 6, 8], [4, 1], 'float32');
$parameter = CudaArray::fromFlatArray([0.5], [1, 1], 'float32')->requiresGrad();
$optimizer = Optimizer::adamW(0.1, 0.9, 0.999, 1e-8, 0.0);
$state = $optimizer->initState([$parameter]);
$initialLoss = null;
$finalLoss = null;

for ($step = 0; $step < 20; $step++) {
    $optimizer->zeroGrad([$parameter]);
    $prediction = $x->matmul($parameter);
    $difference = $prediction->subtract($target);
    $loss = $difference->multiply($difference)->mean();
    $lossValue = (float)$loss->item();
    $initialLoss ??= $lossValue;
    $finalLoss = $lossValue;

    $loss->backward();
    $gradient = $parameter->grad();
    if ($gradient === null) {
        failTraining('AdamW returned no gradient');
    }
    if ($gradient->dtype() !== 'float32' || $gradient->getShape() !== [1, 1]) {
        failTraining('gradient dtype or shape mismatch');
    }
    if (!is_finite((float)$gradient->item())) {
        failTraining('gradient is not finite');
    }

    $result = $optimizer->step([$parameter], [$gradient], $state);
    $parameter = $result['parameters'][0]->requiresGrad();
    $state = $result['state'];
}

$optimizer->zeroGrad([$parameter]);
$finalCheck = $x->matmul($parameter)->subtract($target);
$finalCheck->multiply($finalCheck)->mean()->backward();
$snapshotA = $parameter->grad();
$snapshotB = $parameter->grad();
if ($snapshotA === null || $snapshotB === null || $snapshotA === $snapshotB) {
    failTraining('grad() did not return independent snapshots');
}
$snapshotValue = (float)$snapshotA->item();
$parameter->zeroGrad();
if ($parameter->grad() !== null) {
    failTraining('zeroGrad() did not clear the accumulated gradient');
}
if ((float)$snapshotA->item() !== $snapshotValue) {
    failTraining('gradient snapshot was invalidated by zeroGrad()');
}

if ($initialLoss === null || $finalLoss >= $initialLoss * 0.25) {
    failTraining("loss did not decrease sufficiently: $initialLoss -> $finalLoss");
}

echo "AdamW training passed\n";
echo "dtype=" . $parameter->dtype() . "\n";
echo "steps=20\n";
echo "grad snapshots independent\n";
echo "zeroGrad clears gradient\n";
?>
--EXPECT--
AdamW training passed
dtype=float32
steps=20
grad snapshots independent
zeroGrad clears gradient
