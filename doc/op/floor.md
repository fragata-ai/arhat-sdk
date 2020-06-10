
# Floor

Element-wise application of the floor function to the input tensor `x`. 
Output tensor shape is the same as the input tensor.

Supported types: *float*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Output tensor

## Constructors

### Operator


```
FloorOp(x *core.Tensor, y *core.Tensor) core.Operator

FloorGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Floor(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Floor(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.FloorOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Floor(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Floor(x)
    m.Return(y, 0)
}
```

