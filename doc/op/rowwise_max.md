
# RowwiseMax

Compute row-wise max reduction of the input tensor. This op takes one input, `x`, 
of shape `(B, M, N)`, where `B` is the batch size, `M` is number of rows, and `N` is number 
of columns. The output of this op, `y`, is a matrix of shape `(B, M)`, with one row for each 
element of the batch, and the same number of columns as the number of rows of the input tensor.

Supported types: *float*.

## Interface

### Input

**x**

>A tensor of dimensions `(B, M, N)` to compute rowwise-max. Here, `B` is batch size,  and `M` and `N` are the number of rows and columns of each element of the batch,  respectively.


### Output

**y**

>The output tensor of shape `(B, M)`, where each row represents the row-wise maximums  for that element of the input batch


## Constructors

### Operator


```
RowwiseMaxOp(x *core.Tensor, y *core.Tensor) core.Operator

RowwiseMaxGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) RowwiseMax(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func RowwiseMax(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.RowwiseMaxOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func RowwiseMax(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.RowwiseMax(x)
    m.Return(y, 0)
}
```

