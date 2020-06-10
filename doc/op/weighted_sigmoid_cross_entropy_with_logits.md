
# WeightedSigmoidCrossEntropyWithLogits

Given three matrices: logits, targets, weights, all of the same shape,
`(batchSize, numClasses)`, computes the weighted sigmoid cross entropy between
logits and targets. Specifically, at each position `(r, c)`, this computes
`weights[r, c] * crossentropy(sigmoid(logits[r, c]), targets[r, c])`, and then
averages over each row.
Returns a tensor of shape `(batchSize)` of losses for each example.

Supported types: *float*.

## Interface

### Input

**logits**

>Matrix of logits for each example and class

**targets**

>Matrix of targets, same shape as logits

**weights**

>Matrix of weights, same shape as logits

### Output

**out**

>Vector with the total xentropy for each example

## Constructors

### Operator


```
WeightedSigmoidCrossEntropyWithLogitsOp(
    logits *core.Tensor,
    targets *core.Tensor,
    weights *core.Tensor,
    out *core.Tensor) core.Operator

WeightedSigmoidCrossEntropyWithLogitsGradientOp(
    gradOut *core.Tensor,
    logits *core.Tensor,
    targets *core.Tensor,
    weights *core.Tensor,
    gradLogits *core.Tensor) core.Operator
```


### Layer


```
func(f *Fragment) WeightedSigmoidCrossEntropyWithLogits(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func WeghtedSigmoidCrossEntropyWithLogits(g api.Graph, t core.Type, shape []int) {
    logits := g.NewTensor(t)
    targets := g.NewTensor(t)
    weights := g.NewTensor(t)
    out := g.NewTensor(t)
    g.External(logits, shape)
    g.External(targets, shape)
    g.External(weights, shape)
    g.WeightedSigmoidCrossEntropyWithLogitsOp(logits, targets, weights, out)
    g.Return(out)
}
```


### Layer


```
import "fragata/arhat/front/models"

func WeghtedSigmoidCrossEntropyWithLogits(m *models.Model, t string, shape []int) {
    logits := m.External("dtype", t, "shape", shape)
    targets := m.External("dtype", t, "shape", shape)
    weights := m.External("dtype", t, "shape", shape)
    out := m.WeightedSigmoidCrossEntropyWithLogits(logits, targets, weights)
    m.Return(out, 0)
}
```

