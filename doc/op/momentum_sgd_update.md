
# MomentumSgdUpdate

Performs a momentum SGD update for an input gradient and momentum
parameters. Concretely, given inputs `(param, moment, grad, lr)` and arguments
`(momentum, nesterov)`, computes:

    if not nesterov
        adjustedGradient = lr * grad + momentum * moment
        param = param - adjustedGradient
        outputParam = param
        outputMoment = adjustedGradient
        outputGrad = adjustedGradient
    else
        newMoment = momentum * moment + lr * grad
        param = param - ((1 + momentum) * newMoment - momentum * moment)
        outputParam = (1 + momentum) * newMoment - momentum * moment
        outputMoment = newMoment
        outputGrad = param

Output is `(outputParam, outputMoment, outputGrad)`.

Supported types: *float*.

## Interface

### Input

**param**

>Parameters to be updated

**moment**

>Moment history

**grad**

>Gradient computed

**lr**

>Learning rate

### Output

**outputParam**

>Updated parameter

**outputMoment**

>Updated momentum

**outputGrad**

>Adjusted gradient

### Attributes

**momentum**

>*(type: float; default: 0)* Momentum hyperparameter

**nesterov**

>*(type: bool; default: true)* Whether to use Nesterov Accelerated Gradient

## Constructors

### Operator


```
MomentumSgdUpdateOp(
    param *core.Tensor,
    moment *core.Tensor,
    grad *core.Tensor,
    lr *core.Tensor,
    outputParam *core.Tensor,
    outputMoment *core.Tensor,
    outputGrad *core.Tensor,
    momentum float32,
    nesterov bool) core.Operator
```

