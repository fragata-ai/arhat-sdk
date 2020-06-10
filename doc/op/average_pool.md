
# AveragePool

Consumes an input tensor and applies average pooling across the the tensor according
to kernel sizes, stride sizes, pad lengths and dilation. Average pooling consists
of taking the average value of a subset of the input tensor according to the kernel
size and downsampling the data into the output tensor for further processing.

Pooling layers reduce the spatial dimensionality of the input tensor. Each of the
output tensor's dimensions will reduce according to:

    dim_out = (dim_in - kernel + 2 * pad) / stride + 1

Supported types: *float*.

## Interface

### Input

**x**

>Input data tensor

### Output

**y**

>Output data tensor

### Attributes

**globalPooling**

>*(type: bool; default: false)* If set to *true*, perform global pooling. In case of global pooling, dilation and stride shouldn't be set.


**kernel**

>*(type: []int)* Size of the window to take an average over. Kernel must be explicitly specified (no default value).


**dilation**

>*(type: []int; default: nil)* Parameter that controls the stride of elements in the window. If left at default the dilation will be set to 1.


**stride**

>*(type: []int; default: nil)* Stride of the window. If left at default the stride will be set to 1.


**pads**

>*(type: []int: default: nil)* Implicit zero padding to be added on both sides. If left at default the pads will be set to 0.


**legacyPad**

>*(type: enum; default: "notSet")* Legacy padding strategy, one of *"notSet"*, *"valid"*  or *"same"*. (DEPRECATED).


**countIncludePad**

>*(type: bool; default: false)* When *true*, will include the zero-padding in the averaging


## Constructors

### Operator


```
AveragePoolOp(
    x *core.Tensor,
    y *core.Tensor,
    globalPooling bool,
    kernel []int,
    dilation []int,
    stride []int,
    pads []int,
    legacyPad base.LegacyPadding,
    countIncludePad bool) core.Operator

AveragePoolGradientOp(
    x *core.Tensor,
    y *core.Tensor,
    dy *core.Tensor,
    dx *core.Tensor,
    globalPooling bool,
    kernel []int,
    dilation []int,
    stride []int,
    pads []int,
    legacyPad base.LegacyPadding,
    countIncludePad bool) core.Operator
```


### Layer


```
func(f *Fragment) AveragePool(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
    "fragata/arhat/op/base"
)

func AveragePool(
        g api.Graph, 
        t core.Type, 
        shape []int,
        globalPooling bool,
        kernel []int,
        dilation []int,
        stride []int,
        pads []int,
        countIncludePad bool) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.AveragePoolOp(
        x, 
        y, 
        globalPooling, 
        kernel, 
        dilation, 
        stride, 
        pads, 
        base.LegacyPaddingNotSet, 
        countIncludePad)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func AveragePool(
        m *models.Model, 
        t string, 
        shape []int,
        globalPooling bool,
        kernel []int,
        dilation []int,
        stride []int,
        pads []int,
        countIncludePad bool) {
    x := m.External("dtype", t, "shape", shape)
    y := 
        m.AveragePool(
            x,
            "globalPooling", globalPooling,
            "kernel", kernel,
            "dilation", dilation,
            "stride", stride,
            "pads", pads,
            "countIncludePad", countIncludePad)
    m.Return(y, 0)
}
```

