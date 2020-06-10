
# Scale

Takes an input tensor and produces an output tensor whose value is 
the input tensor scaled element-wise.

Supported types: *float*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Output tensor

### Attributes

**scale**

>*(type: float; default: 1.0)* the scale to apply

## Constructors

### Operator


```
ScaleOp(x *core.Tensor, y *core.Tensor, scale float32) core.Operator

ScaleGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor,
    scale float32) core.Operator
```


### Layer


```
func(f *Fragment) Scale(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Scale(g api.Graph, t core.Type, shape []int, scale float32) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.ScaleOp(x, y, scale)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Scale(m *models.Model, t string, shape []int, scale float32) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Scale(x, "scale", scale)
    m.Return(y, 0)
}
```

