
# PadImage

Pads values around the boundary of an image according to the pad
values and stride sizes. Currently does not support custom strides 
(all strides are fixed to 1).

Supported types: *float*.

## Interface

### Input

**x**

>Input data tensor from the previous operator; the input has size `(N, C, H, W)`, where `N` is the batch size,  `C` is the number of channels, and `H` and `W` are the height and the width of the data.


### Output

**y**

>Output data tensor from padding the `H` and `W` dimensions on the tensor.  Dimensions will vary based on various pad and stride sizes


### Attributes

**pads**

>*(type: []int; default: nil)* Controls the amount of padding to apply to the input image. If left at default the pads will be set to 1.


**mode**

>*(type: enum; default: "constant")* padding mode, one of *"constant"*: pad constant values;  *"reflect"*: pad with reflect values; *"edge"*: pad with the edge values


**value**

>*(type: float; default: 0.0)* padding value to be used in *"constant"* mode


## Constructors

### Operator


```
PadImageOp(
    x *core.Tensor,
    y *core.Tensor,
    pads []int,
    mode base.PadMode,
    value float32) core.Operator

PadImageGradientOp(
    dy *core.Tensor,
    dx *core.Tensor,
    pads []int,
    mode base.PadMode) core.Operator
```


### Layer


```
func(f *Fragment) PadImage(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
    "fragata/arhat/op/base"
)

func PadImage(
        g api.Graph, 
        t core.Type, 
        shape []int, 
        pads []int, 
        mode base.PadMode, 
        value float32) {
    x := g.NewTensor(t) 
    y := g.NewTensor(t) 
    g.External(x, shape)
    g.PadImageOp(x, y, pads, mode, value)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func PadImage(m *models.Model, t string, shape []int, pads []int, mode string, value float32) {
    x := m.External("dtype", t, "shape", shape)
    y := m.PadImage(x, "pads", pads, "mode", mode, "value", value)
    m.Return(y, 0)
}
```

