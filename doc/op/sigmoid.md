
# Sigmoid

Apply the sigmoid function element-wise to the input tensor. This is often used
as a non-linear activation function in a neural network. The sigmoid function is
defined as:

    y = 1 / (1 + EXP(-x))

Supported types: *float*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Output tensor

## Constructors

### Operator


```
SigmoidOp(x *core.Tensor, y *core.Tensor) core.Operator

SigmoidGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Sigmoid(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Sigmoid(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.SigmoidOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Sigmoid(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Sigmoid(x)
    m.Return(y, 0)
}
```

