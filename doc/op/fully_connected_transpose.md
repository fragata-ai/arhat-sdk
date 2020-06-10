
# FullyConnectedTranspose

Same as FullyConnected, but weight matrix is supposed to be already pretransposed.

Supported types: *float, half*.

## Interface

### Input

**x**

>Input tensor to be coerced into a 2D matrix of shape `(M, K)`,  where `M` is the batch size and `K` is the number of features in a single observation


**w**

>Input tensor to be coerced into a 2D matrix of shape `(N, K)` describing a fully connected  weight matrix. Here, `K` is the number of features in a single observation and `N` is  the number of nodes in the FC layer.


**b**

>Input tensor containing vector of length `N` which describes one bias for each node  in the layer


### Output

**y**

>Output tensor containing a 2D output matrix of shape `(M, N)`, where `M` is the batch size and `N` is the number of nodes in the layer. The output is calculated as `y = x * w^T + b`.


### Attributes

**axis**

>*(type: int; default: 1)* Describes the axis of the input data `x`.  Defaults to one because in the common case when the input `X` has shape `(M, K)`,  the first axis encodes the batch size.


**axisW**

>*(type: int; default: 1)* Describes the axis of the input weight matrix `w`.  Defaults to one because the first axis most likely describes the batch size.


## Constructors

### Operator


```
FullyConnectedTransposeOp(
    x *core.Tensor,
    w *core.Tensor,
    b *core.Tensor,
    y *core.Tensor,
    axis int,
    axisW int) core.Operator

FullyConnectedTransposeGradientOp(
    x *core.Tensor,
    w *core.Tensor,
    dy *core.Tensor,
    b *core.Tensor,
    dx *core.Tensor,
    dw *core.Tensor,
    db *core.Tensor,
    axis int,
    axisW int) core.Operator
```


### Layer


```
func(f *Fragment) FullyConnectedTranspose(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func FullyConnectedTranpose(
        g api.Graph, 
        t core.Type, 
        xShape []int,
        wShape []int,
        bShape []int,
        axis int,
        axisW int) {
    x := g.NewTensor(t)
    w := g.NewTensor(t)
    b := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, xShape)
    g.External(w, wShape)
    g.External(b, bShape)
    g.FullyConnectedTransposeOp(x, w, b, y, axis, axisW)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func FullyConnectedTranspose(
        m *models.Model, 
        t string, 
        xShape []int,
        wShape []int,
        bShape []int,
        axis int,
        axisW int) {
    x := m.External("dtype", t, "shape", xShape)
    w := m.External("dtype", t, "shape", wShape)
    b := m.External("dtype", t, "shape", bShape)
    y := m.FullyConnectedTranspose(x, w, b, "axis", axis, "axisW", axisW)
    m.Return(y, 0)
}
```

