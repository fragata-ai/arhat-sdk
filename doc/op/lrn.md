
# Lrn

Applies Local Response Normalization to an input tensor. This operation performs
a kind of "lateral inhibition" by normalizing over local input regions, where
normalization is applied across channels. This operator is typically used to
normalize an unbounded activation (such as ReLU). The output shape is the same as
the input shape.

The formula for LRN is as follows:

    y[c] = x[c] * (bias + alpha / size * (SUM[c1 IN [MAX(0, c-size/2), MIN(N-1, c+size/2)]] x[c1]^2)) ^ (-beta)     

Supported types: *float, half*.

## Interface

### Input

**x**

>Input data tensor (ReLU output)

### Output

**y**

>Output tensor

### Attributes

**size**

>*(type: int; default: 0)* Amount of neighboring channels to sum over for normalization

**alpha**

>*(type: float; default: 0)* Multiplicative (scaling) factor

**beta**

>*(type: float; default: 0)* Exponent

**bias**

>*(type: float; default: 1.0)* Additive factor

## Constructors

### Operator


```
LrnOp(
    x *core.Tensor,
    y *core.Tensor,
    size int,
    alpha float32,
    beta float32,
    bias float32) core.Operator

LrnGradientOp(
    x *core.Tensor,
    y *core.Tensor,
    dy *core.Tensor,
    dx *core.Tensor,
    size int,
    alpha float32,
    beta float32,
    bias float32) core.Operator
```


### Layer


```
func(f *Fragment) Lrn(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Lrn(
        g api.Graph, 
        t core.Type, 
        shape []int, 
        size int, 
        alpha float32, 
        beta float32, 
        bias float32) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.LrnOp(x, y, size, alpha, beta, bias)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Lrn(
        m *models.Model, 
        t string, 
        shape []int, 
        size int, 
        alpha float32, 
        beta float32, 
        bias float32) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Lrn(x, "size", size, "alpha", alpha, "beta", beta, "bias", bias)
    m.Return(y, 0)
}
```

