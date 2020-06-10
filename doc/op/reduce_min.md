
# ReduceMin

Computes the **min** of the input tensor's element along the provided axes.
The resulted tensor has the same rank as the input if `keepdims` equals *true*.
If `keepdims` is set to *false*, then the resulted tensor has the reduced dimension pruned.

Supported types: *int32, int64, float, double*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Reduced output tensor

### Attributes

**axes**

>*(type: []int, default: nil)* A list of integers, along which to reduce


**keepDim**

>*(type: bool, default: true)* Keep the reduced dimension(s) or not,  default *true* keeps the reduced dimension(s)


## Constructors

### Operator


```
ReduceMinOp(
    x *core.Tensor,
    y *core.Tensor,
    axes []int,
    keepDims bool) core.Operator

ReduceMinGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor,
    axes []int) core.Operator
```


### Layer


```
func(f *Fragment) ReduceMin(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func ReduceMin(g api.Graph, t core.Type, shape []int, axes []int, keepDim bool) {
    x := g.NewTensor(t) 
    y := g.NewTensor(t) 
    g.External(x, shape)
    g.ReduceMinOp(x, y, axes, keepDim)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func ReduceMin(m *models.Model, t string, shape []int, axes []int, keepDim bool) {
    x := m.External("dtype", t, "shape", shape)
    y := m.ReduceMin(x, "axes", axes, "keepDim", keepDim)
    m.Return(y, 0)
}
```

