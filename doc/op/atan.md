
# Atan

Calculates the arctangent of the given input tensor, element-wise.

Supported types: *float*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>The arctangent of the input tensor computed element-wise

## Constructors

### Operator


```
AtanOp(x *core.Tensor, y *core.Tensor) core.Operator

AtanGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Atan(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Atan(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.AtanOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Atan(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Atan(x)
    m.Return(y, 0)
}
```

