
# Adam

Computes the Adam update (https://arxiv.org/abs/1412.6980) for an
input gradient and momentum parameters. Concretely, given inputs
`(param, moment1, moment2, grad, lr, iters)`, computes

    t = iters + 1
    correctionMultiplier = sqrt(1 - power(beta2, t)) / (1 - power(beta1, t))
    outputMoment1 = (beta1 * moment1) + (1 - beta1) * grad
    outputMoment2 = (beta2 * moment2) + (1 - beta2) * square(grad)
    outputGrad = correctionMultiplier * outputMoment1 / (sqrt(outputMoment2) + epsilon)
    outputParam = param + lr * outputGrad

and returns `(outputParam, outputMoment1, outputMoment2, outputGrad)`, 
in which outputGrad is an optional output.

Implicit input `iters` containing iteration number is passed by the framework internally.

Supported types: *float*.

## Interface

### Input

**param**

>Parameters to be updated

**moment1**

>First moment history

**moment2**

>Second moment history

**grad**

>Gradient computed

**lr**

>Learning rate

### Output

**outputParam**

>Updated parameters

**outputMoment1**

>Updated first moment

**outputMoment2**

>Updated second moment

**outputGrad**

>Optional effective gradient

### Attributes

**beta1**

>*(type: float; default: 0.9)*

**beta2**

>*(type: float; default: 0.999)*

**epsilon**

>*(type: float; default: 1e-5)*

## Constructors

### Operator


```
AdamOp(
    param *core.Tensor,
    moment1 *core.Tensor,
    moment2 *core.Tensor,
    grad *core.Tensor,
    lr *core.Tensor,
    outputParam *core.Tensor,
    outputMoment1 *core.Tensor,
    outputMoment2 *core.Tensor,
    outputGrad *core.Tensor,
    beta1 float32,
    beta2 float32,
    epsilon float32) core.Operator
```

