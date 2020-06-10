
# HalfToFloat

Casts the elements of a given input tensor of data type *half* to a data type *float*
and returns an output tensor of the same size in the converted type. 

Supported types: *half*.

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
HalfToFloatOp(x *core.Tensor, y *core.Tensor) core.Operator

HalfToFloatGradientOp(dy *core.Tensor, dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) HalfToFloat(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func HalfToFloat(g api.Graph, shape []int) {
    x := g.NewTensor(core.TypeHalf)
    y := g.NewTensor(core.TypeFloat)
    g.External(x, shape)
    g.HalfToFloatOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func HalfToFloat(m *models.Model, shape []int) {
    x := m.External("dtype", "half", "shape", shape)
    y := m.HalfToFloat(x)
    m.Return(y, 0)
}
```

