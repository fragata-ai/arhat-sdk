
# ReduceMean

Computes the **mean** of the input tensor's elements along the provided `axes`. 
The resulting tensor has the same rank as the input if the `keepdims` argument equals *true* (default). 
If `keepdims` is set to *false*, then the `axes` dimensions are pruned.

Supported types: *float*.

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
ReduceMeanOp(
    x *core.Tensor,
    y *core.Tensor,
    axes []int,
    keepDims bool) core.Operator

ReduceMeanGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor,
    axes []int) core.Operator
```


### Layer


```
func(f *Fragment) ReduceMean(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func ReduceMean(g api.Graph, t core.Type, shape []int, axes []int, keepDim bool) {
    x := g.NewTensor(t) 
    y := g.NewTensor(t) 
    g.External(x, shape)
    g.ReduceMeanOp(x, y, axes, keepDim)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func ReduceMean(m *models.Model, t string, shape []int, axes []int, keepDim bool) {
    x := m.External("dtype", t, "shape", shape)
    y := m.ReduceMean(x, "axes", axes, "keepDim", keepDim)
    m.Return(y, 0)
}
```

