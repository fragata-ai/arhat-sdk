
# Adadelta

Computes the AdaDelta update (https://arxiv.org/abs/1212.5701) for an input
gradient and accumulated history of squared gradients. Concretely, given
inputs `(param, momentGrad, momentDelta, grad, lr)`, computes:

    outputMomentGrad = momentGrad * decay + square(grad) * (1 - decay)
    outputGrad = sqrt(momentDelta + epsilon) / sqrt(outputMomentGrad + epsilon) * grad
    outputParam = param + lr * outputGrad
    outputMomentDelta = momentDelta * decay + square(outputGrad) * (1 - decay)

and returns `(outputParam, outputMomentGrad, outputMomentDelta)`.

Supported types: *float*.

## Interface

### Input

**param**

>Parameters to be updated

**momentGrad**

>Average of squared gradients

**momentDelta**

>Average of squared parameter updates

**grad**

>Gradient computed

**lr**

>Learning rate

### Output

**outputParam**

>Updated parameters

**outputMomentGrad**

>Updated average squared gradient

**outputMomentDelta**

>Updated average of squared parameter updates

### Attributes

**epsilon**

>*(type: float; default: 1e-5)*

**decay**

>*(type: float; default: 0.95)* The squared gradient sum is decayed by this factor

## Constructors

### Operator


```
AdadeltaOp(
    param *core.Tensor,
    momentGrad *core.Tensor,
    momentDelta *core.Tensor,
    grad *core.Tensor,
    lr *core.Tensor,
    outputParam *core.Tensor,
    outputMomentGrad *core.Tensor,
    outputMomentDelta *core.Tensor,
    epsilon float32,
    decay float32) core.Operator
```

