
# Sinh

Calculates the hyperbolic sine of the given input tensor, element-wise.

Supported types: *float*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>The hyperbolic sine values of the input tensor, computed element-wise

## Constructors

### Operator


```
SinhOp(x *core.Tensor, y *core.Tensor) core.Operator

SinhGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Sinh(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Sinh(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.SinhOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Sinh(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Sinh(x)
    m.Return(y, 0)
}
```

