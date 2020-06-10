
# Reshape

Reshapes the tensor.

At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is going to be copied
from the input tensor.

For empty tensor, we will set the -1 dimension to be 0 (if one dimension is -1).
When the tensor is empty, dimension of 0 will remain 0.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Reshaped output tensor

### Attributes

**shape**

>*(type: []int; default: nil)* New shape

## Constructors

### Operator


```
ReshapeOp(x *core.Tensor, y *core.Tensor, shape []int) core.Operator

ReshapeGradientOp(
    dy *core.Tensor, 
    x *core.Tensor, 
    y *core.Tensor, 
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Reshape(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Reshape(g api.Graph, t core.Type, shape []int, newShape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.ReshapeOp(x, y, newShape)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Reshape(m *models.Model, t string, shape []int, newShape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Reshape(x, "shape", newShape)
    m.Return(y, 0)
}
```

