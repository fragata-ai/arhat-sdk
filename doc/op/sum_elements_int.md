
# SumElementsInt

Sums the integer elements of the input tensor.

Supported types: *int32*.

## Interface

### Input

**x**

>Tensor to sum up

### Output

**y**

>Scalar tensor containing the sum

## Constructors

### Operator


```
SumElementsIntOp(x *core.Tensor, y *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) SumElementsInt(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func SumElementsInt(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.SumElementsIntOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func SumElementsInt(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.SumElementsInt(x)
    m.Return(y, 0)
}
```

