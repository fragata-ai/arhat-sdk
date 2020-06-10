
# Max

Element-wise max of an arbitrary number of input tensors. All inputs
must have the same shape and data type, and the output will have the same shape
as the inputs.

Supported types: *float*.

## Interface

### Input

**x**

>Input tensors with the same shape

### Output

**y**

>Output tensor with same dimensions as input(s). Contains the maximum valued element at each location.


## Constructors

### Operator


```
MaxOp(x []*core.Tensor, y *core.Tensor) core.Operator

MaxGradientOp(
    y *core.Tensor,
    dy *core.Tensor,
    x []*core.Tensor,
    dx []*core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Max(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Max(g api.Graph, t core.Type, shape []int, count int) {
    x := make([]*core.Tensor, count)
    for i := 0; i < count; i++ {
        x[i] = g.NewTensor(t)
        g.External(x[i], shape)
    }
    y := g.NewTensor(t)
    g.MaxOp(x, y)
    g.Return(y)
}
```


### Layer


```
import (
    "fragata/arhat/front/core"
    "fragata/arhat/front/models"
)

func Max(m *models.Model, t string, shape []int, count int) {
    x := make([]core.Layer, count)
    for i := 0; i < count; i++ {
        x[i] = m.External("dtype", t, "shape", shape)
    }
    y := m.Max(x)
    m.Return(y, 0)
}
```

