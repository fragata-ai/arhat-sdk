
# Elu

Implements the exponential linear unit (ELU) activation function as described in 
[Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289). 
Takes an input tensor `x` of arbitrary shape, computes the elementwise ELU operation, 
and returns a vector `y` of the same shape as output. 
The alpha parameter may be passed as an argument, but defaults to 1. 
The ELU operation is defined as

    y = IF x < 0 THEN alpha * (EXP(x) - 1) ELSE x 

Supported types: *float*.

## Interface

### Input

**x**

>Input tensor of data to be operated on

### Output

**y**

>Output tensor, calculated as described above

### Attributes

**alpha**

>*(type: float; default: 1.0)* Defines alpha parameter used in calculation

## Constructors

### Operator


```
EluOp(x *core.Tensor, y *core.Tensor, alpha float32) core.Operator

EluGradientOp(
    dy *core.Tensor,
    x *core.Tensor,
    y *core.Tensor,
    dx *core.Tensor,
    alpha float32) core.Operator
```


### Layer


```
func(f *Fragment) Elu(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Elu(g api.Graph, t core.Type, shape []int, alpha float32) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.EluOp(x, y, alpha)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Elu(m *models.Model, t string, shape []int, alpha float32) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Elu(x, "alpha", alpha)
    m.Return(y, 0)
}
```

