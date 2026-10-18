--TEST--
CUDA causal attention audit covers masked softmax, backward, NN and Fusion
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;
use PhpTensors\NN;

function attentionFail(string $message): never
{
    throw new RuntimeException($message);
}

function closeEnough(float $actual, float $expected, float $epsilon = 1e-5): bool
{
    return abs($actual - $expected) <= $epsilon;
}

$q = CudaArray::fromFlatArray([1, 0, 0, 1, 1, 1], [3, 2], 'float32')->requiresGrad();
$k = CudaArray::fromFlatArray([1, 0, 0, 1, 1, 1], [3, 2], 'float32');
$v = CudaArray::fromFlatArray([1, 2, 3, 4, 5, 6], [3, 2], 'float32')->requiresGrad();
$mask = CudaArray::fromFlatArray([
    0, -INF, -INF,
    0, 0, -INF,
    0, 0, 0,
], [3, 3], 'float32');
$finiteMask = CudaArray::fromFlatArray([
    0, -1e9, -1e9,
    0, 0, -1e9,
    0, 0, 0,
], [3, 3], 'float32');

$scores = $q->matmul($k->transpose());
$probabilities = $scores->detach()->add($mask)->softmax(-1);
$p = $probabilities->toArray();
for ($row = 0; $row < 3; $row++) {
    if (!closeEnough(array_sum($p[$row]), 1.0)) {
        attentionFail('causal softmax row does not sum to one');
    }
    for ($column = $row + 1; $column < 3; $column++) {
        if ($p[$row][$column] !== 0.0) {
            attentionFail('future position received nonzero probability');
        }
    }
}

$literalAutograd = $scores->add($mask)->softmax(-1);
$literalAutogradValues = $literalAutograd->toArray();
for ($row = 0; $row < 3; $row++) {
    for ($column = $row + 1; $column < 3; $column++) {
        if ($literalAutogradValues[$row][$column] !== 0.0) {
            attentionFail('Autograd masked position received nonzero probability');
        }
    }
}

$finiteProbabilities = $scores->add($finiteMask)->softmax(-1);
$context = $finiteProbabilities->matmul($v);
$context->sum()->backward();
foreach ([$q->grad(), $v->grad()] as $gradient) {
    if ($gradient === null || $gradient->getShape() !== [3, 2]) {
        attentionFail('attention gradient shape mismatch');
    }
    foreach ($gradient->toArray() as $row) {
        foreach ($row as $value) {
            if (!is_finite((float)$value)) {
                attentionFail('attention gradient is not finite');
            }
        }
    }
}

$allMasked = CudaArray::fromFlatArray([-INF, -INF, -INF], [3])->softmax()->toArray();
if (!closeEnough($allMasked[0], 1 / 3) || !closeEnough($allMasked[1], 1 / 3) ||
    !closeEnough($allMasked[2], 1 / 3)) {
    attentionFail('fully masked slice did not use the documented uniform policy');
}

$nn = NN::softmax(CudaArray::fromFlatArray([1000, 999], [1, 2, 1, 1]))->toArray();
if (!closeEnough($nn[0][0][0][0] + $nn[0][1][0][0], 1.0)) {
    attentionFail('NN softmax did not normalize channels');
}

$fusionScores = CudaArray::fromFlatArray([1, 0, 0, 1, 1, 1], [3, 2], 'float32');
$fused = Fusion::run(fn() => $fusionScores->matmul($k->transpose())->add($mask)->softmax(-1))->toArray();
for ($row = 0; $row < 3; $row++) {
    for ($column = 0; $column < 3; $column++) {
        if (!closeEnough((float)$fused[$row][$column], (float)$p[$row][$column])) {
            attentionFail('Fusion softmax differs from eager masked softmax');
        }
    }
}

echo "causal mask: future probabilities zero\n";
echo "eager backward: finite gradients\n";
echo "infinite mask with Autograd: accepted for partial masks\n";
echo "fully masked slice: uniform\n";
echo "NN softmax: channel-wise\n";
echo "Fusion: matches eager\n";
?>
--EXPECT--
causal mask: future probabilities zero
eager backward: finite gradients
infinite mask with Autograd: accepted for partial masks
fully masked slice: uniform
NN softmax: channel-wise
Fusion: matches eager
