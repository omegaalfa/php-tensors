# Migração da API `Cuda` para `PhpTensors`

Este guia descreve a migração do fork antigo para a base experimental atual.
A API pública nova usa `PhpTensors` e não registra aliases para o namespace
antigo `Cuda`.

```php
// Antes
use Cuda\CudaArray;

// Agora
use PhpTensors\CudaArray;
```

## Diferenças de chamadas

```php
// randn antigo: média e desvio eram argumentos públicos
$x = CudaArray::randn([1024], 0.0, 1.0, 'float32');

// randn novo: normal padrão; seed opcional
$x = CudaArray::randn([1024], 'float32', seed: 123);

// rand novo: seed opcional
$u = CudaArray::rand([1024], 0, 1, 'float32', seed: 123);
```

`randn()` não recebe mais `mean`/`std`; aplique as transformações depois da
geração. `rand()` sem seed usa uma seed implícita, sem prometer unicidade
absoluta ou reprodutibilidade entre processos.

`minimum()` e `maximum()` são métodos de instância. A forma antiga estática
torna-se:

```php
$z = $a->minimum($b);
$z = $a->maximum($b);
```

`arange()` infere o dtype quando `$dtype` é `null` (`int64` para sequências
inteiras e `float64` para sequências com ponto flutuante). `linspace()` usa
`float64` por padrão.

O fork antigo usava `clip($minimum, $maximum)`. A base também possui
`clamp($min = null, $max = null)`; `clip()` mantém os dois limites obrigatórios
e não altera a semântica de `clamp()`.

## Autograd

`grad()` retorna um snapshot independente na GPU, não um alias mutável de
`tensor_t::grad`; ele permanece válido depois de `zeroGrad()` ou da destruição
do tensor original. `zeroGrad()` remove o gradiente acumulado e uma consulta
posterior a `grad()` retorna `null`. Nenhum dos dois altera os valores do tensor
ou `requiresGrad()`.

`softmaxDerivative()` recebe valores já normalizados e calcula somente
`y * (1 - y)`, a diagonal da Jacobiana, e não o backward completo de softmax.

## Limitações

- `HostArray` expõe os nomes necessários para paridade de API, mas as operações
  CUDA recuperadas são explicitamente não suportadas no backend CPU.
- FP16/BF16 dependem do suporte efetivo de cada operação; algumas promovem para
  `float32` e outras rejeitam dtypes de baixa precisão.
- Fusion rejeita explicitamente operações recuperadas sem caminho seguro.
- `CudaArray::softmax()` eager trata NaN, múltiplos `+Inf` e fatias somente de
  `-Inf` com política especial. Fatias parcialmente mascaradas com valores
  finitos e `-Inf` seguem a decomposição estável e permitem backward completo,
  com probabilidades e gradientes zero nas posições mascaradas. Fatias somente
  de `-Inf`, NaN e `+Inf` continuam rejeitadas no Autograd. O caminho Fusion
  mantém a decomposição finita existente e não aplica automaticamente essa
  política.
- `NN::softmax()` é separado e ainda não herda a correção especial de infinitos.
- Autograd só cobre operações com regra de backward registrada.

## Argumentos nomeados

Na base atual, use `$dtype`/`$seed` em `randn()`, `$min`/`$max`/`$seed` em
`rand()`, `$minimum`/`$maximum` em `randomUniform()` e `clip()`, e `$other` em
`minimum()`/`maximum()`. `arange()` usa `$start`, `$stop`, `$step`, `$dtype`;
`linspace()` usa `$start`, `$stop`, `$num`, `$endpoint`, `$dtype`.

Chamadas nomeadas escritas para os nomes do fork antigo devem ser ajustadas;
não há alias de assinatura para preservar esses nomes.
