
# ReluN

Takes an input tensor and produces an output tensor where the rectified linear function, 
`y = MIN(MAX(0, x), n)`, is applied to the tensor elementwise.

Supported types: *float*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Output tensor with same shape as input

### Attributes

**n**

>*(type: float; default: 6.0)* The cap of output

## Constructors

### Operator


```
ReluNOp(x *core.Tensor, y *core.Tensor, n float32) core.Operator

ReluNGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor,
    n float32) core.Operator
```


### Layer


```
func(f *Fragment) ReluN(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func ReluN(g api.Graph, t core.Type, shape []int, n float32) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.ReluNOp(x, y, n)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func ReluN(m *models.Model, t string, shape []int, n float32) {
    x := m.External("dtype", t, "shape", shape)
    y := m.ReluN(x, "n", n)
    m.Return(y, 0)
}
```

