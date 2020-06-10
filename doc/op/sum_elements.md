
# SumElements

Sums the elements of the input tensor. Tensor type must be *float*.

Supported types: *float*.

## Interface

### Input

**x**

>Tensor to sum up

### Output

**y**

>Scalar tensor containing the sum or average

### Attributes

**average**

>*(type: bool; default: false)*: Set to *true* to compute the average of the elements  rather than the sum


## Constructors

### Operator


```
SumElementsOp(x *core.Tensor, y *core.Tensor, average bool) core.Operator

SumElementsGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    dx *core.Tensor,
    average bool) core.Operator
```


### Layer


```
func(f *Fragment) SumElements(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func SumElements(g api.Graph, t core.Type, shape []int, average bool) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.SumElementsOp(x, y, average)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func SumElements(m *models.Model, t string, shape []int, average bool) {
    x := m.External("dtype", t, "shape", shape)
    y := m.SumElements(x, "average", average)
    m.Return(y, 0)
}
```

