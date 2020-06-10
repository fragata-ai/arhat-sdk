
# Asin

Calculates the arcsine of the given input tensor, element-wise.

Supported types: *float*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>The arcsine of the input tensor computed element-wise

## Constructors

### Operator


```
AsinOp(x *core.Tensor, y *core.Tensor) core.Operator

AsinGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Asin(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Asin(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.AsinOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Asin(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Asin(x)
    m.Return(y, 0)
}
```

