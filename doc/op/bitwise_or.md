
# BitwiseOr

Performs element-wise operation **bitwise or** with broadcast support.

Supported types: *bool, int32, int64*.

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
BitwiseOrOp(
    a *core.Tensor,
    b *core.Tensor,
    c *core.Tensor,
    legacyBroadcast bool,
    axis int) core.Operator
```


### Layer


```
func(f *Fragment) BitwiseOr(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func BitwiseOr(g api.Graph, t core.Type, aShape []int, bShape []int) {
    a := g.NewTensor(t)
    b := g.NewTensor(t)
    c := g.NewTensor(t)
    g.External(a, aShape)
    g.External(b, bShape)
    g.BitwiseOrOp(a, b, c, false, -1)
    g.Return(c)
}
```


### Layer


```
import "fragata/arhat/front/models"

func BitwiseOr(m *models.Model, t string, aShape []int, bShape []int) {
    a := m.External("dtype", t, "shape", aShape)
    b := m.External("dtype", t, "shape", bShape)
    c := m.BitwiseOr(a, b)
    m.Return(c, 0)
}
```

