
# Mul

Performs element-wise binary multiplication with broadcast support

Supported types: *int32, int64, float, double*.

## Interface

### Input

**a**

>First operand

**b**

>Second operand

### Output

**c**

>Output tensor

### Attributes

**legacyBroadcast**

>*(type: bool, default: false)* When set, the second tensor can either be of size 1  (a scalar value), or having its shape as a contiguous subset of the first tensor's shape.  The starting of the mutually equal shape is specified by the argument `axis`,  or if it is set to -1 (default), suffix matching is assumed.


**axis**

>*(type: int, default: -1)* Axis to be used with legacy broadcasting


## Constructors

### Operator


```
MulOp(
    a *core.Tensor,
    b *core.Tensor,
    c *core.Tensor,
    legacyBroadcast bool,
    axis int) core.Operator

MulGradientOp(
    dc *core.Tensor,
    a *core.Tensor,
    b *core.Tensor,
    c *core.Tensor,
    da *core.Tensor,
    db *core.Tensor,
    legacyBroadcast bool,
    axis int) core.Operator
```


### Layer


```
func(f *Fragment) Mul(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Mul(g api.Graph, t core.Type, aShape []int, bShape []int) {
    a := g.NewTensor(t)
    b := g.NewTensor(t)
    c := g.NewTensor(t)
    g.External(a, aShape)
    g.External(b, bShape)
    g.MulOp(a, b, c, false, -1)
    g.Return(c)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Mul(m *models.Model, t string, aShape []int, bShape []int) {
    a := m.External("dtype", t, "shape", aShape)
    b := m.External("dtype", t, "shape", bShape)
    c := m.Mul(a, b)
    m.Return(c, 0)
}
```

