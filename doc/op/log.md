
# Log

Calculates the natural logarithm of the given input tensor, element-wise.

Supported types: *float*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Output tensor computed as the natural logarithm of the input tensor computed, element-wise

## Constructors

### Operator


```
LogOp(x *core.Tensor, y *core.Tensor) core.Operator

LogGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Log(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Log(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.LogOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Log(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Log(x)
    m.Return(y, 0)
}
```

