
# Cos

Calculates the cosine of the given input tensor, element-wise.

Supported types: *float*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Output tensor calculated as the cosine of the input tensor, element-wise

## Constructors

### Operator


```
CosOp(x *core.Tensor, y *core.Tensor) core.Operator

CosGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Cos(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Cos(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.CosOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Cos(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Cos(x)
    m.Return(y, 0)
}
```

