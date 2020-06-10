
# Split

Split an `input` tensor into a list of tensors, along the axis specified by the `axis` dimension. 
The lengths of the split can be specified using argument `split`. 
Otherwise, the tensor is split to equal sized parts.

## Interface

### Input

**input**

>tensor to split

### Output

**output**

>output tensors

### Attributes

**axis**

>(*int*) axis to split on

**split**

>(*[]int*) length of each output

## Constructors

### Operator


```
SplitOp(
    input *core.Tensor,
    output []*core.Tensor,
    axis int,
    split []int) core.Operator

SplitGradientOp(
    gradOutput []*core.Tensor,
    gradInput *core.Tensor,
    axis int) core.Operator
```


### Layer


```
func(f *Fragment) Split(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Split(g api.Graph, t core.Type, shape []int, axis int, split []int) {
    x := g.NewTensor(t)
    count := len(split)
    y := make([]*core.Tensor, count)
    for i := 0; i < count; i++ {
        y[i] = g.NewTensor(t)
    }
    g.External(x, shape)
    g.SplitOp(x, y, axis, split)
    for i := 0; i < count; i++ {
        g.Return(y[i])
    }
}
```


### Layer


```
import "fragata/arhat/front/models"

func Split(m *models.Model, t string, shape []int, axis int, split []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Split(x, "axis", axis, "split", split)
    count := len(split)
    for i := 0; i < count; i++ {
        m.Return(y, i)
    }
}
```

