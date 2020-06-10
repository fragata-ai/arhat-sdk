
# Adagrad

Computes the AdaGrad update for an input gradient and accumulated
history. Concretely, given inputs `(param, grad, moment, lr)`,
computes

    outputMoment = moment + square(grad)
    effectiveLr = lr / (sqrt(outputMoment) + epsilon)
    update = lr * grad / (sqrt(outputMoment) + epsilon)
    outputParam = param + update

and returns `(outputParam, outputMoment)`.

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

>Updated parameters

**outputMoment**

>Updated moment

### Attributes

**epsilon**

>*(type: float; default: 1e-5)*

**decay**

>*(type: float; default: 1.0)* If it is in (0, 1), the gradient square sum is decayed by this factor


## Constructors

### Operator


```
AdagradOp(
    param *core.Tensor, 
    moment *core.Tensor, 
    grad *core.Tensor, 
    lr *core.Tensor, 
    outputParam *core.Tensor, 
    outputMoment *core.Tensor, 
    epsilon float32,
    decay float32) core.Operator
```

