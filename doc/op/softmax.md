
# Softmax

Applies the Softmax function to an n-dimensional input Tensor rescaling them so
that the elements of the n-dimensional output Tensor lie in the range (0, 1) and
sum to 1. The softmax operator is typically the last layer in a classifier network,
as its output can be interpreted as confidence probabilities of an input belonging
to each class. The input is a 2D tensor of size (batchSize x
inputFeatureDimensions). The output tensor has the same shape and contains the
softmax normalized values of the corresponding input. The softmax function is
defined as follows:

    y[i] = EXP(x[i]) / (SUM[j] EXP(x[j]))

The input does not need to explicitly be a 2D vector; rather, it will be coerced
into one. For an arbitrary n-dimensional tensor `x` with dimensions
`(a[0], a[1], ..., a[k-1], a[k], ..., a[n-1])`, if `k` is the `axis` provided,
then `x` will be coerced into a 2D tensor with dimensions
`(a[0] * ... * a[k-1], a[k] * ... * a[n-1])`. For the default case where
`axis = 1`, the `x` tensor will be coerced into a 2D tensor of dimensions
`(a[0], a[1] * ... * a[n-1])`, where `a[0]` is often the batch size. In this
situation, we must have `a[0] = N` and `a[1] * ... * a[n-1] = D`. Each of these
dimensions must be matched correctly, or else the operator will throw errors.

Supported types: *float, half*.

## Interface

### Input

**x**

>Input tensor that's coerced into a 2D matrix of size `(N, D)` as described above

### Output

**y**

>The softmax normalized output tensor with the same shape as input tensor

### Attributes

**axis**

>*(type: int; default: 1)* Axis of the inputs when coerced to 2D matrix

## Constructors

### Operator


```
SoftmaxOp(x *core.Tensor, y *core.Tensor, axis int) core.Operator

SoftmaxGradientOp(
    y *core.Tensor,
    dy *core.Tensor,
    dx *core.Tensor,
    axis int) core.Operator
```


### Layer


```
func(f *Fragment) Softmax(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Softmax(g api.Graph, t core.Type, shape []int, axis int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.SoftmaxOp(x, y, axis)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Softmax(m *models.Model, t string, shape []int, axis int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Softmax(x, "axis", axis)
    m.Return(y, 0)
}
```

