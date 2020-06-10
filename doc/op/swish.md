
# Swish

Takes an input tensor and produces an output tensor where the swish function, 
`y = x / (1 + EXP(-x))`, is applied to the tensor elementwise.

Supported types: *float*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>output tensor

## Constructors

### Operator


```
SwishOp(x *core.Tensor, y *core.Tensor) core.Operator

SwishGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Swish(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Swish(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.SwishOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Swish(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Swish(x)
    m.Return(y, 0)
}
```

