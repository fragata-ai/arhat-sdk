
# Softplus

Takes one input data tensor `x` and produces one output data tensor `y,` 
where the softplus function, `y = LOG(EXP(x) + 1)`, is applied to `x` elementwise.

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
SoftplusOp(x *core.Tensor, y *core.Tensor) core.Operator

SoftplusGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Softplus(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Softplus(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.SoftplusOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Softplus(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Softplus(x)
    m.Return(y, 0)
}
```

