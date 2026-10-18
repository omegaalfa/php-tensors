--TEST--
CUDA Fusion Autograd and AdamW agree with eager training
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;
use PhpTensors\Optimizer;

function fusionTrainingFail(string $message): never
{
    throw new RuntimeException($message);
}

function closeTraining(float $actual, float $expected, float $epsilon = 1e-4): void
{
    if (!is_finite($actual) || !is_finite($expected) ||
        abs($actual - $expected) > $epsilon) {
        fusionTrainingFail("expected $expected, got $actual");
    }
}

function trainingLoss(CudaArray $x, CudaArray $target, CudaArray $parameter): float
{
    $difference = $x->matmul($parameter)->subtract($target);
    return (float)$difference->multiply($difference)->mean()->item();
}

$x = CudaArray::fromFlatArray([1, 2, 3, 4], [4, 1], 'float32');
$target = CudaArray::fromFlatArray([2, 4, 6, 8], [4, 1], 'float32');

// The same forward/backward calculation is evaluated eagerly and in Fusion.
$eagerProbe = CudaArray::fromFlatArray([0.5], [1, 1], 'float32')->requiresGrad();
$eagerLoss = $x->matmul($eagerProbe)->subtract($target);
$eagerLoss = $eagerLoss->multiply($eagerLoss)->mean();
$eagerLossValue = (float)$eagerLoss->item();
$eagerLoss->backward();
$eagerGradient = $eagerProbe->grad();

$fusedProbe = CudaArray::fromFlatArray([0.5], [1, 1], 'float32')->requiresGrad();
$fusedResult = Fusion::run(function () use ($x, $target, $fusedProbe) {
    $difference = $x->matmul($fusedProbe)->subtract($target);
    $loss = $difference->multiply($difference)->mean();
    $loss->backward();
    return [$loss, $fusedProbe->grad()];
});
$fusedLossValue = (float)$fusedResult[0]->item();
$fusedGradient = $fusedResult[1];
if ($eagerGradient === null || $fusedGradient === null) {
    fusionTrainingFail('eager or Fusion backward produced no gradient');
}
closeTraining($fusedLossValue, $eagerLossValue);
closeTraining((float)$fusedGradient->item(), (float)$eagerGradient->item());
if ($fusedGradient->dtype() !== 'float32' ||
    $fusedGradient->getShape() !== [1, 1]) {
    fusionTrainingFail('Fusion gradient dtype or shape mismatch');
}

// Verify independent gradient snapshots and zeroGrad semantics.
$snapshotA = $eagerProbe->grad();
$snapshotB = $eagerProbe->grad();
if ($snapshotA === null || $snapshotB === null || $snapshotA === $snapshotB) {
    fusionTrainingFail('grad() snapshots are not independent');
}
$snapshotValue = (float)$snapshotA->item();
$eagerProbe->zeroGrad();
if ($eagerProbe->grad() !== null || (float)$snapshotA->item() !== $snapshotValue) {
    fusionTrainingFail('zeroGrad invalidated the gradient snapshot');
}

$optimizer = Optimizer::adamW(0.1, 0.9, 0.999, 1e-8, 0.0);
$eagerParameter = CudaArray::fromFlatArray([0.5], [1, 1], 'float32')->requiresGrad();
$eagerState = $optimizer->initState([$eagerParameter]);
$eagerInitial = trainingLoss($x, $target, $eagerParameter);
$eagerFinal = $eagerInitial;
for ($step = 0; $step < 20; $step++) {
    $optimizer->zeroGrad([$eagerParameter]);
    $difference = $x->matmul($eagerParameter)->subtract($target);
    $loss = $difference->multiply($difference)->mean();
    $loss->backward();
    $gradient = $eagerParameter->grad();
    if ($gradient === null || !is_finite((float)$gradient->item())) {
        fusionTrainingFail('eager AdamW gradient is not finite');
    }
    $stepResult = $optimizer->step([$eagerParameter], [$gradient], $eagerState);
    $eagerParameter = $stepResult['parameters'][0]->requiresGrad();
    $eagerState = $stepResult['state'];
    $eagerFinal = trainingLoss($x, $target, $eagerParameter);
}

// Compile one Fusion graph and replay it with the updated parameter/state.
$fusedParameter = CudaArray::fromFlatArray([0.5], [1, 1], 'float32')->requiresGrad();
$fusedState = $optimizer->initState([$fusedParameter]);
$graph = Fusion::compile(function (
    CudaArray $parameter,
    CudaArray $firstMoment,
    CudaArray $secondMoment,
    CudaArray $beta1Power,
    CudaArray $beta2Power,
    CudaArray $input,
    CudaArray $expected
) use ($optimizer) {
    $optimizer->zeroGrad([$parameter]);
    $difference = $input->matmul($parameter)->subtract($expected);
    $loss = $difference->multiply($difference)->mean();
    $loss->backward();
    $gradient = $parameter->grad();
    $step = $optimizer->step(
        [$parameter], [$parameter->grad()], [
            'firstMoment' => [$firstMoment],
            'secondMoment' => [$secondMoment],
            'beta1Power' => $beta1Power,
            'beta2Power' => $beta2Power,
        ]
    );
    $step['gradient'] = $gradient;
    return $step;
}, [
    $fusedParameter,
    $fusedState['firstMoment'][0],
    $fusedState['secondMoment'][0],
    $fusedState['beta1Power'],
    $fusedState['beta2Power'],
    $x,
    $target,
]);

$fusedInitial = trainingLoss($x, $target, $fusedParameter);
$fusedFinal = $fusedInitial;
for ($step = 0; $step < 20; $step++) {
    $oldParameter = $fusedParameter;
    $stepResult = $graph->run(
        $fusedParameter,
        $fusedState['firstMoment'][0],
        $fusedState['secondMoment'][0],
        $fusedState['beta1Power'],
        $fusedState['beta2Power'],
        $x,
        $target
    );
    $graphGradient = $stepResult['gradient'];
    if ($graphGradient === null || !is_finite((float)$graphGradient->item())) {
        fusionTrainingFail('Fusion AdamW gradient is not finite');
    }
    $fusedParameter = $stepResult['parameters'][0]->requiresGrad();
    $fusedState = $stepResult['state'];
    $fusedFinal = trainingLoss($x, $target, $fusedParameter);
}

if ($eagerFinal >= $eagerInitial * 0.25 ||
    $fusedFinal >= $fusedInitial * 0.25) {
    fusionTrainingFail('AdamW loss did not decrease sufficiently');
}
closeTraining($fusedFinal, $eagerFinal, 2e-3);
closeTraining((float)$fusedParameter->item(), (float)$eagerParameter->item(), 2e-3);

echo "eager/Fusion forward and backward agree\n";
echo "gradient snapshots and zeroGrad: passed\n";
echo "AdamW Fusion replay: 20 steps\n";
echo "eager and Fusion losses decreased\n";
echo "finite gradients and parameters\n";
?>
--EXPECT--
eager/Fusion forward and backward agree
gradient snapshots and zeroGrad: passed
AdamW Fusion replay: 20 steps
eager and Fusion losses decreased
finite gradients and parameters
