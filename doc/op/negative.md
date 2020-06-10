
# Negative

Performs element-wise negation on the input tensor.

Supported types: *int32, int64, float, double*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Negated output tensor

## Constructors

### Operator


```
NegativeOp(x *core.Tensor, y *core.Tensor) core.Operator

NegativeGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Negative(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Negative(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.NegativeOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Negative(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Negative(x)
    m.Return(y, 0)
}
```

