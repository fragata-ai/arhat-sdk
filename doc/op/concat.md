
# Concat

Concatenate a list of tensors into a single tensor.
The `axis` argument specifies an axis along which the arrays will be concatenated.
When *true* (default is *false*), the `addAxis` argument adds the axis specified in 
`axis` to all input tensors.

## Interface

### Input

**input**

>Input tensors

### Output

**output**

>Concatenated tensor

### Attributes

**axis**

>The axis along which the arrays will be concatenated


**addAxis**

>When *true*, the axis specified in `axis` is added to all input tensors


## Constructors

### Operator


```
ConcatOp(
    input []*core.Tensor,
    output *core.Tensor,
    axis int,
    addAxis bool) core.Operator

ConcatGradientOp(
    gradOutput *core.Tensor,
    gradInput []*core.Tensor,
    axis int,
    addAxis bool) core.Operator
```


### Layer


```
func(f *Fragment) Concat(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Concat(g api.Graph, t core.Type, shape [][]int, axis int, addAxis bool) {
    n := len(shape)
    input := make([]*core.Tensor, n)
    for i := 0; i < n; i++ {
        input[i] = g.NewTensor(t)
        g.External(input[i], shape[i])
    }
    output := g.NewTensor(t)
    g.ConcatOp(input, output, axis, addAxis)
    g.Return(output)
}
```


### Layer


```
import (
    "fragata/arhat/front/core"
    "fragata/arhat/front/models"
)

func Concat(m *models.Model, t string, shape [][]int, axis int, addAxis bool) {
    n := len(shape)
    input := make([]core.Layer, n)
    for i := 0; i < n; i++ {
        input[i] = m.External("dtype", t, "shape", shape[i])
    }
    output := m.Concat(input, "axis", axis, "addAxis", addAxis)
    m.Return(output, 0)
}
```

