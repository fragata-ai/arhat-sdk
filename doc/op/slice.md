
# Slice

Produces a slice of the input tensor.

- Currently, only slicing in a single dimension is supported.

- Start and end indices are passed using the `starts` and `ends` arguments.

- If a negative value is passed for any of the start or end indices, it represents 
  the number of elements before the end of that dimension. End indices are non-inclusive 
  unless negative (end index -1 means up to and including the last element).

## Interface

### Input

**x**

>Tensor to extract slices from

### Output

**y**

>Sliced output tensor

### Attributes

**starts**

>*(type: []int;  default: nil)* list of starting indices

**ends**

>*(type: []int;  default: nil)* list of ending indices

## Constructors

### Operator


```
SliceOp(
    x *core.Tensor,
    y *core.Tensor,
    starts []int,
    ends []int) core.Operator

SliceGradientOp(
    x *core.Tensor,
    dy *core.Tensor,
    dx *core.Tensor,
    starts []int,
    ends []int) core.Operator
```


### Layer


```
func(f *Fragment) Slice(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Slice(g api.Graph, t core.Type, shape []int, starts []int, ends []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.SliceOp(x, y, starts, ends)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Slice(m *models.Model, t string, shape []int, starts []int, ends []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Slice(x, "starts", starts, "ends", ends)
    m.Return(y, 0)
}
```

