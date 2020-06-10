
# FullyConnected

Computes an output `(y)` as a linear combination of the input data tensor `(x)` with 
a weight tensor `(w)` and bias tensor `(b)`. More formally,

    y = x * w^T + b

Here, `x` is a matrix of shape `(M, K)`, `w` is a matrix of shape `(N, K)`, 
`b` is a vector of length `N`, and `y` is a matrix of shape `(M, N)`. 
`N` can be thought of as the number of nodes in the layer, `M` is the batch size, 
and `K` is the number of features in an input observation.

NOTE: `x` does not need to explicitly be a 2-dimensional matrix, however, if it is not 
it will be coerced into one. For an arbitrary `n`-dimensional tensor `x`, 
with dimensions `(a[0], a[1], ... , a[k-1], a[k], ... , a[n-1])`, 
if `k` is the `axis` arg provided, then `x` will be coerced into a 2-dimensional tensor 
with dimensions `(a[0] * ... * a[k-1], a[k] * ... * a[n-1])`. For the default case 
where `axis = 1`, this means the `x` tensor will be coerced into a 2D tensor of dimensions 
`(a[0], a[1] * ... * a[n-1])`, where `a[0]` is often the batch size. In this situation, 
we must have `a[0] = M` and `a[1] * ... * a[n-1] = K`. Lastly, even though `b` is a vector
of length `N`, it is copied and resized to shape `(M, N)` implicitly, then added to 
each vector in the batch.

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
FullyConnectedOp(
    x *core.Tensor,
    w *core.Tensor,
    b *core.Tensor,
    y *core.Tensor,
    axis int,
    axisW int) core.Operator

FullyConnectedGradientOp(
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
func(f *Fragment) FullyConnected(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func FullyConnected(
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
    g.FullyConnectedOp(x, w, b, y, axis, axisW)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func FullyConnected(
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
    y := m.FullyConnected(x, w, b, "axis", axis, "axisW", axisW)
    m.Return(y, 0)
}
```

