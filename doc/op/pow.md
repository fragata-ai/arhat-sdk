
# Pow

Takes an input data tensor `x` and an exponent parameter `exponent`, which can be a scalar or 
another tensor. As output, it produces a single output data tensor `y`, where the function 
`f(x) = x ^ exponent` has been applied to `x` elementwise.

Supported types: *float*.

## Interface

### Input

**a**

>Input data tensor to be operated on

**b**

>Exponent tensor containing the exponent(s) for calculation. This input is optional; set to *nil* is setting exponent via attribute.


### Output

**c**

>Output data tensor with the same shape as the input

### Attributes

**exponent**

>*(type: float; default: 0.0) The exponent of the power function.  Do not use if setting exponent via input.


**enableBroadcast**

>*(type: bool; default: false)* Set to *true* to enable broadcast. Broadcst may be enabled only if exponent is specified via input tensor.


**axis**

>*(type: int; default: -1)* broadcast axis to be used if broadcasting is enabled


## Constructors

### Operator


```
PowOp(
    a *core.Tensor,
    b *core.Tensor,
    c *core.Tensor,
    exponent float32,
    enableBroadcast bool,
    axis int) core.Operator

PowGradientOp(
    dc *core.Tensor,
    a *core.Tensor,
    b *core.Tensor,
    c *core.Tensor,
    da *core.Tensor,
    db *core.Tensor,
    exponent float32,
    enableBroadcast bool,
    axis int) core.Operator
```


### Layer


```
func(f *Fragment) Pow(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Pow(
        g api.Graph, 
        t core.Type, 
        aShape []int, 
        bShape []int, 
        exponent float32, 
        enableBroadcast bool, 
        axis int) {
    a := g.NewTensor(t)
    var b *core.Tensor
    if len(bShape) != 0 {
        b = g.NewTensor(t)
    }
    c := g.NewTensor(t)    
    g.External(a, aShape)
    if len(bShape) != 0 {
        g.External(b, bShape)
    }
    g.PowOp(a, b, c, exponent, enableBroadcast, axis)
    g.Return(c)
}
```


### Layer


```
import (
    "fragata/arhat/front/core"
    "fragata/arhat/front/models"
)

func Pow(
        m *models.Model, 
        t string, 
        aShape []int, 
        bShape []int, 
        exponent float32, 
        enableBroadcast bool, 
        axis int) {
    a := m.External("dtype", t, "shape", aShape)
    var b core.Layer
    if len(bShape) != 0 {
        b = m.External("dtype", t, "shape", bShape)
    }
    c := m.Pow(a, b, "exponent", exponent, "enableBroadcast", enableBroadcast, "axis", axis)
    m.Return(c, 0)
}
```

