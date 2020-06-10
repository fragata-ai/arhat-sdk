
# Conv

Computes a convolution operation over an input tensor `(x)`, with a filter tensor `(w)`
and a bias tensor `(b)`, and outputs a single output tensor `(y)`. 
The input `(x)` is a tensor of shape `(N, C_in, H_in, W_in)` and the output `(y)` is 
a tensor of shape `(N, C_out, H_out, W_out)`. Here, `N` is the batch size, 
`C` is the number of channels, `H` is the spatial height, and `W` is the spatial width. 
For example, if your input data was a batch of five 100 x 120 pixel RGB images, 
`x` would have shape `(5, 3, 120, 100)`.

The `w` input tensor may contain multiple filters and has shape `(M, C_in, K_H, K_W)`. 
Here, `M` is the number of individual filters contained in the tensor, `C_in` is
the number of channels of each filter (by convention in 2D convolution it is the same as
the number of channels in the input), `K_H` is the spatial height of the kernel, and `K_W` is
the spatial width of the kernel. The `b` tensor is a vector of length `M`, where there is 
one bias for each filter in the `w` tensor.

Given the shape of the input tensor and the filter tensor, we can calculate the shape of 
the output tensor as follows. The number of items in the batch `N` will stay the same. 
The number of channels in the output will equal the number of kernels in the filter tensor, 
so `C_out = M.` With stride and pad defined below, the spatial height and width of
the output (`H_out` and `W_out`) are calculated as

    H_out = (H_in - K_H + 2 * pad) / stride + 1

    W_out = (W_in - K_W + 2 * pad) / stride + 1

Supported types: *float, half*.

## Interface

### Input

**x**

>Input data tensor, of shape `(N, C_in, H_in, W_in)`, to be convolved with  the kernels in the filter tensor


**w**

>The filter tensor, of shape `(M, C_in, K_H, K_W)`, containing the filters to be  convolved with the data


**b**

>The bias tensor, of length `M`, containing the biases for the convolution,  one bias per filter


### Output

**y**

>Output data tensor, of shape `(N, C_out, H_out, W_out)`, that contains the result of the convolution


### Attributes

**kernel**

>*(type: []int; default: nil)* Desired kernel size.  Kernel must be explicitly specified (no default value).


**dilation**

>*(type: []int; default: nil)* Controls spacing between kernel points.  If dilation is greater than one, the kernel does not operate on a contiguous spatial region.  If left at default the dilation will be set to 1.


**stride**

>*(type: []int; default: nil)* Controls the stride of the kernel as it traverses the input tensor. If left at default the stride will be set to 1.


**pads**

>*(type: []int; default: nil)* Controls the amount of padding to apply to the input feature map before computing the convolution. If left at default the pads will be set to 1.


**legacyPad**

>*(type: enum; default: "notSet")* Legacy padding strategy, one of *"notSet"*, *"valid"*  or *"same"*. (DEPRECATED).


**group**

>*(type: int; default: 1)* Controls level of group convolution.


## Constructors

### Operator


```
ConvOp(
    x *core.Tensor,
    w *core.Tensor,
    b *core.Tensor,
    y *core.Tensor,
    kernel []int,
    dilation []int,
    stride []int,
    pads []int,
    legacyPad base.LegacyPadding,
    group int) core.Operator

ConvGradientOp(
    x *core.Tensor,
    w *core.Tensor,
    dy *core.Tensor,
    dw *core.Tensor,
    dx *core.Tensor,
    db *core.Tensor,
    kernel []int,
    dilation []int,
    stride []int,
    pads []int,
    legacyPad base.LegacyPadding,
    group int) core.Operator
```


### Layer


```
func(f *Fragment) Conv(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
    "fragata/arhat/op/base"
)

func Conv(
        g api.Graph, 
        t core.Type, 
        xShape []int,
        wShape []int,
        bShape []int,
        kernel []int,
        dilation []int,
        stride []int,
        pads []int,
        group int) {
    x := g.NewTensor(t)
    w := g.NewTensor(t)
    b := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, xShape)
    g.External(w, wShape)
    g.External(b, bShape)
    g.ConvOp(x, w, b, y, kernel, dilation, stride, pads, base.LegacyPaddingNotSet, group)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Conv(
        m *models.Model, 
        t string, 
        xShape []int,
        wShape []int,
        bShape []int,
        kernel []int,
        dilation []int,
        stride []int,
        pads []int,
        group int) {
    x := m.External("dtype", t, "shape", xShape)
    w := m.External("dtype", t, "shape", wShape)
    b := m.External("dtype", t, "shape", bShape)
    y := 
        m.Conv(
            x, 
            w, 
            b,
            "kernel", kernel,
            "dilation", dilation,
            "stride", stride,
            "pads", pads,
            "group", group)
    m.Return(y, 0)
}
```

