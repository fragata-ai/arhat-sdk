
# SumSqrElements

Sums the squared elements of the input tensor.

Supported types: *float, half*.

## Interface

### Input

**x**

>Tensor to sum up

### Output

**y**

>Scalar tensor containing the sum of squares

### Attributes

**average**

>*(type: bool; default: false)*: Set to *true* to compute the average  rather than the sum


## Constructors

### Operator


```
SumSqrElementsOp(x *core.Tensor, y *core.Tensor, average bool) core.Operator
```


### Layer


```
func(f *Fragment) SumSqrElements(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func SumSqrElements(g api.Graph, t core.Type, shape []int, average bool) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.SumSqrElementsOp(x, y, average)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func SumSqrElements(m *models.Model, t string, shape []int, average bool) {
    x := m.External("dtype", t, "shape", shape)
    y := m.SumSqrElements(x, "average", average)
    m.Return(y, 0)
}
```

