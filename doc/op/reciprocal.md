
# Reciprocal

Performs element-wise reciprocal (`1 / x`) of input tensor `x`.

Supported types: *float*.

## Interface

### Input

**x**

>Input data tensor

### Output

**y**

>Output tensor

## Constructors

### Operator


```
ReciprocalOp(x *core.Tensor, y *core.Tensor) core.Operator

ReciprocalGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Reciprocal(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Reciprocal(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.ReciprocalOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Reciprocal(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Reciprocal(x)
    m.Return(y, 0)
}
```

