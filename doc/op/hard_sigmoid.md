
# HardSigmoid

Applies hard sigmoid operation to the input data element-wise.
The HardSigmoid operation takes one input `x`, produces one output `y`, and is defined as:

    y = MAX(0, MIN(1, x * alpha + beta))

Supported types: *float*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Output tensor with same shape as input

### Attributes

**alpha**

>*(type: float, default: 0.2)* The slope of the function

**beta**

>*(type: float, default: 0.5)* The bias value of the function

## Constructors

### Operator


```
HardSigmoidOp(
    x *core.Tensor,
    y *core.Tensor,
    alpha float32,
    beta float32) core.Operator

HardSigmoidGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor,
    alpha float32) core.Operator
```


### Layer


```
func(f *Fragment) HardSigmoid(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func HardSigmoid(g api.Graph, t core.Type, shape []int, alpha float32, beta float32) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.HardSigmoidOp(x, y, alpha, beta)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func HardSigmoid(m *models.Model, t string, shape []int, alpha float32, beta float32) {
    x := m.External("dtype", t, "shape", shape)
    y := m.HardSigmoid(x, "alpha", alpha, "beta", beta)
    m.Return(y, 0)
}
```

