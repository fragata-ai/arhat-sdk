
# Logit

Elementwise logit transform: `y = LOG(x / (1 - x))`, where `x` is the
input data clampped in `(eps, 1-eps)`.

Supported types: *float*.

## Interface

### Input

**x**

>Input float tensor

### Output

**y**

>Output float tensor

### Attributes

**eps**

>*(type: float; default: 1e-6)* small positive epsilon value

## Constructors

### Operator


```
LogitOp(x *core.Tensor, y *core.Tensor, eps float32) core.Operator

LogitGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor,
    eps float32) core.Operator
```


### Layer


```
func(f *Fragment) Logit(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Logit(g api.Graph, t core.Type, shape []int, eps float32) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.LogitOp(x, y, eps)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Logit(m *models.Model, t string, shape []int, eps float32) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Logit(x, "eps", eps)
    m.Return(y, 0)
}
```

