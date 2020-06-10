
# Not

Performs element-wise logical negation on input tensor `x`.

Supported types: *bool*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Negated output tensor

## Constructors

### Operator


```
NotOp(x *core.Tensor, y *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Not(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Not(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.NotOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Not(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Not(x)
    m.Return(y, 0)
}
```

