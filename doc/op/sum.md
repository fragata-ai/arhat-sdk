
# Sum

Element-wise sum of each of the input tensors. 
All inputs and outputs must have the same shape and data type.

Supported types: *int32, float*.

## Interface

### Input

**x**

>Input tensors

### Output

**y**

>Output tensor

## Constructors

### Operator


```
SumOp(x []*core.Tensor, y *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Sum(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Sum(g api.Graph, t core.Type, shape []int, count int) {
    x := make([]*core.Tensor, count)
    for i := 0; i < count; i++ {
        x[i] = g.NewTensor(t)
        g.External(x[i], shape)
    }
    y := g.NewTensor(t)
    g.SumOp(x, y)
    g.Return(y)
}
```


### Layer


```
import (
    "fragata/arhat/front/core"
    "fragata/arhat/front/models"
)

func Sum(m *models.Model, t string, shape []int, count int) {
    x := make([]core.Layer, count)
    for i := 0; i < count; i++ {
        x[i] = m.External("dtype", t, "shape", shape)
    }
    y := m.Sum(x)
    m.Return(y, 0)
}
```

