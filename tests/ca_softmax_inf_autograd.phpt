--TEST--
CudaArray softmax supports partial -Inf masks in Autograd
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\InvalidArgumentException;

function softmaxInfFail(string $message): never
{
    throw new RuntimeException($message);
}

function assertClose(float $actual, float $expected, float $epsilon = 2e-4): void
{
    if (!is_finite($actual) || abs($actual - $expected) > $epsilon) {
        softmaxInfFail(sprintf('expected %.8f, got %.8f', $expected, $actual));
    }
}

$values = [1.0, 0.0, -INF, 1.0, 2.0, -INF];
$weights = [0.5, -1.0, 2.0, -0.25, 0.75, 1.5];

$scores = CudaArray::fromFlatArray($values, [2, 3], 'float32')->requiresGrad();
$probabilities = $scores->softmax(-1);
$p = $probabilities->toArray();
if ($p[0][2] !== 0.0 || $p[1][2] !== 0.0) {
    softmaxInfFail('masked probabilities are not exactly zero');
}
assertClose(array_sum($p[0]), 1.0);
assertClose(array_sum($p[1]), 1.0);

$loss = $probabilities
    ->multiply(CudaArray::fromFlatArray($weights, [2, 3], 'float32'))
    ->sum();
$loss->backward();
$gradient = $scores->grad();
if ($gradient === null) softmaxInfFail('missing softmax gradient');
$g = $gradient->toArray();
if ($g[0][2] !== 0.0 || $g[1][2] !== 0.0) {
    softmaxInfFail('masked gradient is not exactly zero');
}
foreach ($g as $row) {
    foreach ($row as $value) {
        if (!is_finite((float)$value)) softmaxInfFail('softmax gradient is not finite');
    }
}

foreach ([0, 1] as $row) {
    $start = $row * 3;
    $rowValues = [$values[$start], $values[$start + 1], 0.0];
    $rowWeights = [$weights[$start], $weights[$start + 1], $weights[$start + 2]];
    $dot = $rowWeights[0] * $p[$row][0] + $rowWeights[1] * $p[$row][1];
    assertClose((float)$g[$row][0], (float)($p[$row][0] * ($rowWeights[0] - $dot)));
    assertClose((float)$g[$row][1], (float)($p[$row][1] * ($rowWeights[1] - $dot)));
}

function weightedLoss(array $values, array $weights): float
{
    $probabilities = CudaArray::fromFlatArray($values, [2, 3], 'float32')->softmax(-1);
    return (float)$probabilities
        ->multiply(CudaArray::fromFlatArray($weights, [2, 3], 'float32'))
        ->sum()->item();
}

$h = 1e-3;
$plus = $values;
$minus = $values;
$plus[0] += $h;
$minus[0] -= $h;
$finiteDifference = (weightedLoss($plus, $weights) - weightedLoss($minus, $weights)) / (2.0 * $h);
assertClose((float)$g[0][0], $finiteDifference, 3e-3);

$viewBase = CudaArray::fromFlatArray([1.0, 9.0, -INF, 8.0, 7.0, -INF], [2, 3], 'float32');
$view = $viewBase->slice(':, ::2');
$viewValues = $view->softmax(-1)->toArray();
if ($viewValues[0][1] !== 0.0 || $viewValues[1][1] !== 0.0) {
    softmaxInfFail('non-contiguous view did not preserve -Inf masking');
}

foreach (['float32', 'float64'] as $dtype) {
    $typed = CudaArray::fromFlatArray($values, [2, 3], $dtype)->softmax(-1)->toArray();
    if ($typed[0][2] !== 0.0 || $typed[1][2] !== 0.0) {
        softmaxInfFail("$dtype masked probability is not zero");
    }
}

foreach ([[-INF, -INF], [NAN, 1.0], [INF, 1.0]] as $invalid) {
    try {
        CudaArray::fromFlatArray($invalid, [2], 'float32')->requiresGrad()->softmax();
        softmaxInfFail('invalid special slice unexpectedly entered Autograd');
    } catch (InvalidArgumentException) {
        // The policy remains explicit for all-masked, NaN and +Inf slices.
    }
}

echo "partial -Inf mask: forward and backward\n";
echo "masked probabilities and gradients: zero\n";
echo "finite-difference check: passed\n";
echo "views and float64: passed\n";
echo "all -Inf, NaN and +Inf Autograd: rejected\n";
?>
--EXPECT--
partial -Inf mask: forward and backward
masked probabilities and gradients: zero
finite-difference check: passed
views and float64: passed
all -Inf, NaN and +Inf Autograd: rejected
