--TEST--
CudaArray softmax handles NaN and infinities per slice
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

function show_softmax(array $values, string $dtype): void {
    $result = CudaArray::fromFlatArray($values, [count($values)], $dtype)->softmax()->toArray();
    foreach ($result as $value) printf("%s ", is_nan($value) ? "NAN" : sprintf("%.6f", $value));
    echo "\n";
}

show_softmax([INF, INF, 0.0, -INF], 'float32');
show_softmax([INF, INF, 0.0, -INF], 'float64');
show_softmax([INF, 0.0, -INF], 'float64');
show_softmax([-INF, -INF], 'float64');
show_softmax([NAN, 1.0], 'float64');

$view = CudaArray::fromFlatArray([
    INF, 0.0, 2.0, -INF,
    INF, INF, 0.0, -INF,
], [2, 4], 'float64')->slice(':, ::2')->softmax(1)->toArray();
printf("%.6f %.6f %.6f %.6f\n", $view[0][0], $view[0][1], $view[1][0], $view[1][1]);

foreach (['float16', 'bfloat16'] as $dtype) {
    try {
        $low = CudaArray::fromFlatArray([INF, INF, 0.0, -INF], [4], $dtype)->softmax()->toArray();
        printf("%s %.6f %.6f\n", $dtype, $low[0], $low[1]);
    } catch (Throwable $error) {
        echo $dtype, " unsupported\n";
    }
}

try {
    CudaArray::fromFlatArray([INF, 0.0], [2], 'float64')->requiresGrad()->softmax();
    echo "autograd missed\n";
} catch (Throwable $error) {
    echo get_class($error), "\n";
}
?>
--EXPECT--
0.500000 0.500000 0.000000 0.000000
0.500000 0.500000 0.000000 0.000000
1.000000 0.000000 0.000000
0.500000 0.500000
NAN NAN
1.000000 0.000000 1.000000 0.000000
float16 0.500000 0.500000
bfloat16 0.500000 0.500000
PhpTensors\InvalidArgumentException
