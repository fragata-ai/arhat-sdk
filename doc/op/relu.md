
# Relu

Applies Rectified Linear Unit (ReLU) operation to the input data element-wise. 
The ReLU operation takes one input `x`, produces one output `y`, and is defined as:

    y = MAX(0, x)

Supported types: *float, half*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Output tensor with same shape as input

## Constructors

### Operator


```
ReluOp(x *core.Tensor, y *core.Tensor) core.Operator

ReluGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Relu(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Relu(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.ReluOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Relu(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Relu(x)
    m.Return(y, 0)
}
```

