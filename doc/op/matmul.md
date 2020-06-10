
# Matmul

Matrix multiplication `y = a * b`, where `a` has size `(M, K)`, `b` has size
`(K, N)`, and `y` will have a size `(M, N)`. To transpose `a` or `b` before
multiplication, pass *true* to the `transA` and/or `transB` arguments, which
separate the first and second dimensions of the respective matrices using
`axisA` and `axisB`.

Supported types: *float*.

## Interface

### Input

**a**

>2D matrix of size `(M, K)`

**b**

>2D matrix of size `(K, N)`

### Output

**y**

>2D matrix of size `(M, N)`

### Attributes

**axisA**

>*(type: int; default: 1)* Exclusive axis that divides the first and second dimension of matrix `a`


**axisB**

>*(type: int; default: 1)* Exclusive axis that divides the first and second dimension of matrix `b`.


**transA**

>*(type: bool; default: false)* Pass *true* to transpose `a` before multiplication and after the dimension adjustment using `axisA`


**transB**

>*(type: bool; default: false)* Pass *true* to transpose `b` before multiplication and after the dimension adjustment using `axisB`


## Constructors

### Operator


```
MatmulOp(
    a *core.Tensor,
    b *core.Tensor,
    y *core.Tensor,
    axisA int,
    axisB int,
    transA bool,
    transB bool) core.Operator

MatmulGradientOp(
    dy *core.Tensor,
    a *core.Tensor,
    b *core.Tensor,
    da *core.Tensor,
    db *core.Tensor,
    axisA int,
    axisB int,
    transA bool,
    transB bool) core.Operator
```


### Layer


```
func(f *Fragment) Matmul(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Matmul(
        g api.Graph, 
        t core.Type, 
        aShape []int, 
        bShape []int, 
        axisA int, 
        axisB int, 
        transA bool, 
        transB bool) {
    a := g.NewTensor(t) 
    b := g.NewTensor(t) 
    y := g.NewTensor(t) 
    g.External(a, aShape)
    g.External(b, bShape)
    g.MatmulOp(a, b, y, axisA, axisB, transA, transB)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Matmul(
        m *models.Model, 
        t string, 
        aShape []int, 
        bShape []int, 
        axisA int, 
        axisB int, 
        transA bool, 
        transB bool) {
    a := m.External("dtype", t, "shape", aShape)
    b := m.External("dtype", t, "shape", bShape)
    y := m.Matmul(a, b, "axisA", axisA, "axisB", axisB, "transA", transA, "transB", transB)
    m.Return(y, 0)
}
```

