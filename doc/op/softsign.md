
# Softsign

Takes one input data tensor `x` and produces one output data `y`, 
where the softsign function, `y = x / (1 + |x|)`, is applied to `x` elementwise.

Supported types: *float*.

## Interface

### Input

**x**

>Input tensor to be operated on

### Output

**y**

>Output tensor with same shape as input

## Constructors

### Operator


```
SoftsignOp(x *core.Tensor, y *core.Tensor) core.Operator

SoftsignGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Softsign(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Softsign(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.SoftsignOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Softsign(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Softsign(x)
    m.Return(y, 0)
}
```

