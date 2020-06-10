
# Lars

Implements Layer-wise Adaptive Rate Scaling (LARS) with clipping. Before adding weight
decay, given a parameter tensor `x` and its gradient `dx`, the local learning rate
for `x` will be

    localLr = trust * norm(x) / (norm(dx) + wd * norm(x) + offset * norm(x))

            = trust / (norm(dx) / norm(x) + wd + offset),

where offset is a preset hyper-parameter to avoid numerical issue and trust
indicates how much we trust the layer to change its parameters during one update.
In this implementation, we use l2 norm and the computed local learning rate is
clipped based on the upper bound `lrMax` and the lower bound `lrMin`:

    localLr = min(localLr, lrMax); localLr = max(localLr, lrMin)

Supported types: *float*.

## Interface

### Input

**x**

>Parameter tensor

**dx**

>Gradient tensor

**wd**

>Weight decay

**trust**

>Trust

**lrMax**

>Upper bound of learning rate

### Output

**lrRescaled**

>Rescaled local learning rate

### Attributes

**offset**

>*(type: float; default: 0.5)* Rescaling offset parameter

**lrMin**

>*(type: float; default: 0.02)* Minimum learning rate for clipping

## Constructors

### Operator


```
LarsOp(
    x *core.Tensor,
    dx *core.Tensor,
    wd *core.Tensor,
    trust *core.Tensor,
    lrMax *core.Tensor,
    lrRescaled *core.Tensor,
    offset float32,
    lrMin float32) core.Operator
```

