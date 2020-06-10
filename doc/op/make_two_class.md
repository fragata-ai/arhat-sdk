
# MakeTwoClass

Given a vector of probabilities, transforms this into a 2-column
matrix with complimentary probabilities for binary classification.

Supported types: *float*.

## Interface

### Input

**x**

>Input vector of probabilities

### Output

**y**

>2-column matrix with complimentary probabilities of `x` for binary classification

## Constructors

### Operator


```
MakeTwoClassOp(x *core.Tensor, y *core.Tensor) core.Operator

MakeTwoClassGradientOp(dy *core.Tensor, dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) MakeTwoClass(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func MakeTwoClass(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.MakeTwoClassOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func MakeTwoClass(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.MakeTwoClass(x)
    m.Return(y, 0)
}
```

