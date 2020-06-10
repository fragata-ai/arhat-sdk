
# Tile

Constructs a tensor by tiling a given tensor along a specified axis. This operation creates
a new tensor by replicating the input tensor a number of times specified by the `tiles` 
argument along the `axis` dimension. The output tensor's `axis` dimension has 
`(x.dims(axis) * tiles)` elements.

Supported types: *int32, int64, float, double*.

## Interface

### Input

**x**

>Input tensor

### Output

**y**

>Output tensor

### Attributes

**tiles**

>*(type: int; default: 1)* Number of replicas

**axis**

>*(type: int; default: 0)* Axis to replicate along

## Constructors

### Operator


```
TileOp(
    x *core.Tensor,
    y *core.Tensor,
    tiles int,
    axis int) core.Operator

TileGradientOp(
    dy *core.Tensor,
    dx *core.Tensor,
    tiles int,
    axis int) core.Operator
```


### Layer


```
func(f *Fragment) Tile(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func Tile(g api.Graph, t core.Type, shape []int, tiles int, axis int) {
    x := g.NewTensor(t)
    y := g.NewTensor(t)
    g.External(x, shape)
    g.TileOp(x, y, tiles, axis)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func Tile(m *models.Model, t string, shape []int, tiles int, axis int) {
    x := m.External("dtype", t, "shape", shape)
    y := m.Tile(x, "tiles", tiles, "axis", axis)
    m.Return(y, 0)
}
```

