--TEST--
CudaArray minimum and maximum preserve NaN and signed-zero semantics
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

function classify(float $value): string {
    if (is_nan($value)) return 'NaN';
    if ($value == 0.0) return (fdiv(1.0, $value) < 0.0) ? '-0' : '+0';
    return (string)$value;
}

function classifyPair(CudaArray $value): string {
    return implode(',', array_map('classify', $value->toArray()));
}

$plus = new CudaArray([0.0], 'float32');
$minus = new CudaArray([-0.0], 'float32');
echo classifyPair($plus->minimum($minus)), "\n";
echo classifyPair($minus->minimum($plus)), "\n";
echo classifyPair($plus->maximum($minus)), "\n";
echo classifyPair($minus->maximum($plus)), "\n";

$nan = new CudaArray([NAN], 'float64');
$one = new CudaArray([1.0], 'float64');
echo classifyPair($nan->minimum($one)), "\n";
echo classifyPair($one->maximum($nan)), "\n";

$a = (new CudaArray([0.0], 'float64'))->requiresGrad();
$b = (new CudaArray([-0.0], 'float64'))->requiresGrad();
$a->minimum($b)->sum()->backward();
echo json_encode([$a->grad()->toArray(), $b->grad()->toArray()]), "\n";

$c = (new CudaArray([0.0], 'float64'))->requiresGrad();
$d = (new CudaArray([-0.0], 'float64'))->requiresGrad();
$c->maximum($d)->sum()->backward();
echo json_encode([$c->grad()->toArray(), $d->grad()->toArray()]), "\n";

$e = (new CudaArray([NAN], 'float64'))->requiresGrad();
$f = (new CudaArray([1.0], 'float64'))->requiresGrad();
$e->minimum($f)->sum()->backward();
echo json_encode([$e->grad()->toArray(), $f->grad()->toArray()]), "\n";

$broadcastLeft = new CudaArray([[2.0, -1.0], [3.0, 4.0]], 'float32');
$broadcastRight = new CudaArray([1.0, 5.0], 'float32');
echo json_encode($broadcastLeft->minimum($broadcastRight)->toArray()), "\n";
$view = (new CudaArray([[0.0, -0.0], [2.0, -3.0]], 'float64'))->transpose();
echo json_encode($view->maximum(new CudaArray([[-1.0], [-2.0]], 'float64'))->toArray()), "\n";

$half = new CudaArray([0.0, -0.0], 'float16');
echo classifyPair($half->minimum(new CudaArray([-0.0, 0.0], 'float16'))), "\n";
$bfloat = new CudaArray([0.0, -0.0], 'bfloat16');
echo classifyPair($bfloat->maximum(new CudaArray([-0.0, 0.0], 'bfloat16'))), "\n";
?>
--EXPECT--
-0
-0
+0
+0
NaN
NaN
[[0],[1]]
[[1],[0]]
[[1],[0]]
[[1,-1],[1,4]]
[[0,2],[-0,-2]]
-0,-0
+0,+0
