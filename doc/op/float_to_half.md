
# FloatToHalf

Casts the elements of a given input tensor of data type *float* to a data type *half*
and returns an output tensor of the same size in the converted type. 

Supported types: *float*.

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
FloatToHalfOp(x *core.Tensor, y *core.Tensor) core.Operator

FloatToHalfGradientOp(dy *core.Tensor, dx *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) FloatToHalf(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func FloatToHalf(g api.Graph, shape []int) {
    x := g.NewTensor(core.TypeFloat)
    y := g.NewTensor(core.TypeHalf)
    g.External(x, shape)
    g.FloatToHalfOp(x, y)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func FloatToHalf(m *models.Model, shape []int) {
    x := m.External("dtype", "float", "shape", shape)
    y := m.FloatToHalf(x)
    m.Return(y, 0)
}
```

