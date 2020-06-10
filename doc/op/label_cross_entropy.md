
# LabelCrossEntropy

Computes the cross entropy between a 2D input data tensor `x` of shape `(N, D)` and 
a 1D input label tensor `label`. The op produces an output tensor `y` of length `N`. 
Here, `N` is considered the batch size and `D` is the size of each 
element in the batch. In practice, it is most commonly used at the end of models as a part of
the loss computation, after the Softmax operator and before the AveragedLoss operator. 
The cross entropy operation is defined as follows

    y[i] = -LOG(x[i, j])

where `(i, j)` is the classifier's prediction of the `j`-th class (the correct one), 
and `i` is the batch size. Each log has a lower limit for numerical stability.

Types supported: *int32* for *label*, *float* otherwise.

## Interface

### Input

**x**

>Input tensor which is almost always the result of a softmax operation.  `x` is a 2D array of shape `(N, D)`, where `N` is the batch size and `D` is the number of classes.


**label**

>Tensor containing the labels used to compare the input. `label` is a length `N`  array of integers, where each element is the integer label for the `n`th element of the batch.


### Output

**y**

>Output tensor from the cross entropy computation. `y` is 1D length `N` tensor.


## Constructors

### Operator


```
LabelCrossEntropyOp(
    x *core.Tensor,
    label *core.Tensor,
    y *core.Tensor) core.Operator

LabelCrossEntropyGradientOp(
    x *core.Tensor,
    label *core.Tensor,
    dy *core.Tensor,
    dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) LabelCrossEntropy(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func LabelCrossEntropy(g api.Graph, t core.Type, xShape []int, labelShape []int) {
    x := g.NewTensor(t)
    label := g.NewTensor(core.TypeInt32)
    y := g.NewTensor(t)
    g.External(x, xShape)
    g.External(label, labelShape)
    g.LabelCrossEntropyOp(x, label, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func LabelCrossEntropy(m *models.Model, t string, xShape []int, labelShape []int) {
    x := m.External("dtype", t, "shape", xShape)
    label := m.External("dtype", "int32", "shape", labelShape)
    y := m.LabelCrossEntropy(x, label)
    m.Return(y, 0)
}
```

