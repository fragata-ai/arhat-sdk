
# WeightedSum

Element-wise weighted sum of several (data, weight) tensor pairs.
Input should be as sequence of tensors `(x[0], weight[0], x[1], weight[1], ...)` where `x[i]` all
have the same shape, and `weight[i]` are size 1 tensors that specifies the weight
of each tensor.

## Interface

### Input

**x**

>Input tensors (even indices) and corresponding weights (odd indices)

### Output

**y**

>Result containing weighted element-wise sum of inputs

## Constructors

### Operator


```
WeightedSumOp(x []*core.Tensor, y *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) WeightedSum(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func WeightedSum(g api.Graph, t core.Type, shape []int, count int) {
    x := make([]*core.Tensor, 2*count)
    for i := 0; i < 2 * count; i++ {
        x[i] = g.NewTensor(t)
    }
    y := g.NewTensor(t)
    one := []int{1}
    for i := 0; i < 2 * count; i += 2 {
        g.External(x[i], shape)
        g.External(x[i+1], one)
    }
    g.WeightedSumOp(x, y)
    g.Return(y)
}
```


### Layer


```
import (
    "fragata/arhat/front/core"
    "fragata/arhat/front/models"
)

func WeightedSum(m *models.Model, t string, shape []int, count int) {
    one := []int{1}
    x := make([]core.Layer, 2*count)
    for i := 0; i < 2 * count; i += 2 {
        x[i] = m.External("dtype", t, "shape", shape)
        x[i+1] = m.External("dtype", t, "shape", one)
    }
    y := m.WeightedSum(x)
    m.Return(y, 0)
}
```

