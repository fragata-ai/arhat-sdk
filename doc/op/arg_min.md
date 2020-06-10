
# ArgMin

Retrieve the argmin of an axis dimension specified by the `axis`
argument. Given an input tensor and two arguments (`axis` and
`keepdims`), returns a tensor containing the indices of the smallest
element along the given axis. If the `keepdims` arg is *true* (default),
the shape of the output tensor matches the input tensor except the
`axis` dimension equals 1. Else, the `axis` dimension of the output
tensor is removed.

Supported types: *int32, int64, float, double*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Tensor of indices for the smallest values

### Attributes

**axis**

>*(type: int; default: -1)* The axis to get argmin


**keepDims**

>*(type: bool; default: true)* If *true* (default), the output tensor shape will match the input tensor shape except the `axis` dimension equals 1. Else, the `axis` dimension of the output tensor is removed.        


## Constructors

### Operator


```
ArgMinOp(
    x *core.Tensor,
    y *core.Tensor,
    axis int,
    keepDims bool) core.Operator
```


### Layer


```
func(f *Fragment) ArgMin(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func ArgMin(g api.Graph, t core.Type, shape []int, axis int, keepDims bool) {
    x := g.NewTensor(t)
    y := g.NewTensor(core.TypeInt32)
    g.External(x, shape)
    g.ArgMinOp(x, y, axis, keepDims)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func ArgMin(m *models.Model, t string, shape []int, axis int, keepDims bool) {
    x := m.External("dtype", t, "shape", shape)
    y := m.ArgMin(x, "axis", axis, "keepDims", keepDims)
    m.Return(y, 0)
}
```

