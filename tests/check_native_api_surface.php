<?php
declare(strict_types=1);

/*
 * Compare the frozen stub surface with a loaded native extension.
 * Run with:
 *   php -n -d extension=/path/phptensors.so tests/check_native_api_surface.php --build=cuda
 */

$build = null;
foreach (array_slice($argv, 1) as $argument) {
    if (str_starts_with($argument, '--build=')) {
        $build = substr($argument, 8);
    } else {
        fwrite(STDERR, "Usage: php -n -d extension=/path/phptensors.so "
            . "tests/check_native_api_surface.php --build=cuda|cpu\n");
        exit(2);
    }
}

if (!in_array($build, ['cuda', 'cpu'], true)) {
    fwrite(STDERR, "Missing or invalid --build=cuda|cpu.\n");
    exit(2);
}

if (!extension_loaded('phptensors')) {
    fwrite(STDERR, "The phptensors extension must be loaded by the caller.\n");
    exit(2);
}

$baselinePath = __DIR__ . '/api_surface.json';
$baseline = json_decode((string)file_get_contents($baselinePath), true, 512, JSON_THROW_ON_ERROR);

function nativeType(?ReflectionType $type): ?string
{
    if ($type === null) {
        return null;
    }

    $value = (string)$type;
    if ($type instanceof ReflectionUnionType) {
        $parts = array_map(static fn(ReflectionType $part): string => (string)$part, $type->getTypes());
        sort($parts);
        $value = implode('|', $parts);
    }

    return $value;
}

function normalizeType(?string $type, string $className): ?string
{
    if ($type === null) {
        return null;
    }
    $type = ltrim($type, '\\');
    $type = str_replace('|' . $className, '|self', $type);
    $type = str_replace($className . '|', 'self|', $type);
    if ($type === $className) {
        return 'self';
    }
    if (str_starts_with($type, '?')) {
        $type = 'null|' . substr($type, 1);
    }
    $parts = explode('|', $type);
    $parts = array_map(static fn(string $part): string => ltrim($part, '\\'), $parts);
    sort($parts);
    return implode('|', array_unique($parts));
}

function nativeParameter(ReflectionParameter $parameter): array
{
    $default = $parameter->isDefaultValueAvailable() ? $parameter->getDefaultValue() : null;
    if (is_int($default) || is_float($default)) {
        $default = (float)$default;
    }
    return [
        'name' => $parameter->getName(),
        'type' => nativeType($parameter->getType()),
        'byReference' => $parameter->isPassedByReference(),
        'variadic' => $parameter->isVariadic(),
        'optional' => $parameter->isOptional(),
        'hasDefault' => $parameter->isDefaultValueAvailable(),
        'default' => $default,
    ];
}

function normalizeDefault(mixed $value): mixed
{
    if (is_int($value) || is_float($value)) {
        return (float)$value;
    }
    if (is_array($value)) {
        foreach ($value as $key => $item) {
            $value[$key] = normalizeDefault($item);
        }
    }
    return $value;
}

function nativeMethods(string $className): array
{
    $class = new ReflectionClass($className);
    $methods = [];
    foreach ($class->getMethods(ReflectionMethod::IS_PUBLIC) as $method) {
        if ($method->getDeclaringClass()->getName() !== $className) {
            continue;
        }
        $methods[$method->getName()] = [
            'static' => $method->isStatic(),
            'abstract' => $method->isAbstract(),
            'returnsReference' => $method->returnsReference(),
            'parameters' => array_map('nativeParameter', $method->getParameters()),
            'returnType' => nativeType($method->getReturnType()),
        ];
    }
    ksort($methods);
    return $methods;
}

$actualClasses = [];
foreach (get_declared_classes() as $className) {
    if (str_starts_with($className, 'PhpTensors\\')) {
        $actualClasses[$className] = nativeMethods($className);
    }
}
ksort($actualClasses);

$missingClasses = [];
$unexpectedClasses = [];
$signatureMismatches = [];
$unsupportedVariadic = [];
$missingMethods = [];

foreach ($baseline['classes'] as $className => $expectedClass) {
    if (!isset($actualClasses[$className])) {
        $missingClasses[] = $className;
        continue;
    }

    $expectedMethods = $expectedClass['methods'];
    $actualMethods = $actualClasses[$className];
    foreach ($expectedMethods as $methodName => $expectedMethod) {
        if (!isset($actualMethods[$methodName])) {
            $missingMethods[] = $className . '::' . $methodName;
            continue;
        }

        $actualMethod = $actualMethods[$methodName];
        foreach ($expectedMethod['parameters'] as &$parameter) {
            $parameter['default'] = normalizeDefault($parameter['default']);
        }
        unset($parameter);
        foreach ($actualMethod['parameters'] as &$parameter) {
            $parameter['default'] = normalizeDefault($parameter['default']);
        }
        unset($parameter);
        $expectedMethod['returnType'] = normalizeType($expectedMethod['returnType'], $className);
        $actualMethod['returnType'] = normalizeType($actualMethod['returnType'], $className);
        foreach ($expectedMethod['parameters'] as &$parameter) {
            $parameter['type'] = normalizeType($parameter['type'], $className);
        }
        unset($parameter);
        foreach ($actualMethod['parameters'] as &$parameter) {
            $parameter['type'] = normalizeType($parameter['type'], $className);
        }
        unset($parameter);

        if ($className === 'PhpTensors\\HostArray' &&
            count($actualMethod['parameters']) === 1 &&
            $actualMethod['parameters'][0]['variadic'] &&
            $actualMethod['parameters'][0]['name'] === 'arguments') {
            $unsupportedVariadic[] = $className . '::' . $methodName;
            continue;
        }

        if ($expectedMethod !== $actualMethod) {
            $signatureMismatches[$className . '::' . $methodName] = [
                'expected' => $expectedMethod,
                'actual' => $actualMethod,
            ];
        }
    }
}

foreach ($actualClasses as $className => $_methods) {
    if (!isset($baseline['classes'][$className])) {
        $unexpectedClasses[] = $className;
    }
}

$expectedClassCount = count($baseline['classes']);
$actualClassCount = count($actualClasses);
$expectedMethodCount = array_sum(array_map(
    static fn(array $class): int => count($class['methods']),
    $baseline['classes']
));
$actualMethodCount = array_sum(array_map('count', $actualClasses));

echo "Build: {$build}\n";
echo "Classes: expected={$expectedClassCount}, actual={$actualClassCount}\n";
echo "Methods: expected={$expectedMethodCount}, actual={$actualMethodCount}\n";

if ($missingClasses) {
    echo "Classes unavailable in this build:\n";
    foreach ($missingClasses as $className) {
        echo "  - {$className}\n";
    }
}
if ($unexpectedClasses) {
    echo "Unexpected native classes:\n";
    foreach ($unexpectedClasses as $className) {
        echo "  - {$className}\n";
    }
}
if ($missingMethods) {
    echo "Methods unavailable in this build:\n";
    foreach ($missingMethods as $method) {
        echo "  - {$method}\n";
    }
}
if ($unsupportedVariadic) {
    echo "HostArray methods registered as variadic unsupported operations:\n";
    foreach ($unsupportedVariadic as $method) {
        echo "  - {$method}\n";
    }
}
if ($signatureMismatches) {
    echo "Signature mismatches:\n";
    foreach ($signatureMismatches as $method => $difference) {
        echo "  - {$method}\n";
        echo "    expected: " . json_encode($difference['expected'], JSON_UNESCAPED_SLASHES) . "\n";
        echo "    actual:   " . json_encode($difference['actual'], JSON_UNESCAPED_SLASHES) . "\n";
    }
}

$differenceCount = count($missingClasses) + count($unexpectedClasses) +
    count($missingMethods) + count($unsupportedVariadic) + count($signatureMismatches);
if ($differenceCount === 0) {
    echo "Native API matches the frozen baseline.\n";
    exit(0);
}

echo "Native API differs from the frozen baseline; no baseline was modified.\n";
exit(1);
