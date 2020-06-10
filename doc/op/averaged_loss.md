
# AveragedLoss

Takes a single 1D input tensor *x* and returns a single output float value *y*. 
The output represents the average of the values in *x*. 
This op is commonly used for averaging losses, hence the name, however it does not exclusively 
operate on losses.

Supported types: *float*.

## Interface

### Input

**x**

>The input tensor

### Output

**y**

>The output tensor of size 1 containing the averaged value

## Constructors

### Operator


```
AveragedLossOp(x *core.Tensor, y *core.Tensor) core.Operator

AveragedLossGradientOp(dy *core.Tensor, x *core.Tensor, dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) AvergedLoss(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func AveragedLoss(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.AveragedLossOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func AveragedLoss(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.AveragedLoss(x)
    m.Return(y, 0)
}
```

