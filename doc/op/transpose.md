
# Transpose

Transpose the input tensor by permuting the axes of the input according
to the `axes` argument.

For example, when `axes` is set to `(1, 0, 2)`, given an input tensor of shape
`(1, 2, 3), the output shape will be `(2, 1, 3).

Supported types: *int32, int64, float, double*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Transposed output

### Attributes

**axes**

>"*(type: []int; default: nil)* Order to permute axes of input tensor.  Reverses the dimensions by default.


## Constructors

### Operator


```
TransposeOp(x *core.Tensor, y *core.Tensor, axes []int) core.Operator

TransposeGradientOp(dy *core.Tensor, dx *core.Tensor, axes []int) core.Operator
```


### Layer


```
func(f *Fragment) Transpose(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Transpose(g api.Graph, t core.Type, shape []int, axes []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.TransposeOp(x, y, axes)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Transpose(m *models.Model, t string, shape []int, axes []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Transpose(x, "axes", axes)
    m.Return(y, 0)
}
```

