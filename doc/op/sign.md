
# Sign

Computes sign for each element of the input: -1, 0 or 1.

Supported types: *int32, int64, float, double*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Output tensor

## Constructors

### Operator


```
SignOp(x *core.Tensor, y *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) Sign(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Sign(g api.Graph, t core.Type, shape []int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.SignOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Sign(m *models.Model, t string, shape []int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Sign(x)
    m.Return(y, 0)
}
```

