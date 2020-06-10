
# ReduceSum

Computes the **sum** of the input tensor's elements along the provided `axes`. 
The resulting tensor has the same rank as the input if the `keepdims` argument equals *true* (default). 
If `keepdims` is set to *false*, then the `axes` dimensions are pruned.

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
ReduceSumOp(
    x *core.Tensor,
    y *core.Tensor,
    axes []int,
    keepDims bool) core.Operator

ReduceSumGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor,
    axes []int) core.Operator
```


### Layer


```
func(f *Fragment) ReduceSum(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func ReduceSum(g api.Graph, t core.Type, shape []int, axes []int, keepDim bool) {
    x := g.NewTensor(t) 
    y := g.NewTensor(t) 
    g.External(x, shape)
    g.ReduceSumOp(x, y, axes, keepDim)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func ReduceSum(m *models.Model, t string, shape []int, axes []int, keepDim bool) {
    x := m.External("dtype", t, "shape", shape)
    y := m.ReduceSum(x, "axes", axes, "keepDim", keepDim)
    m.Return(y, 0)
}
```

