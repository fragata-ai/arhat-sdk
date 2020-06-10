
# SigmoidCrossEntropyWithLogits

Given two matrices logits and targets, of same shape, `(batchSize, numClasses)`, 
computes the sigmoid cross entropy between the two.
Returns a tensor of shape `(batchSize)` of losses for each example.

Supported types: *float*.

## Interface

### Input

**logits**

>Matrix of logits for each example and class

**targets**

>Matrix of targets, same shape as logits

### Output

**out**

>Vector with the total xentropy for each example

### Attributes

**logDTrick**

>*(type: bool, default: false)* If enabled, will use the log d trick to avoid the vanishing gradients early on; see Goodfellow et. al (2014)


**unjoinedLrLoss**

>*(type: bool, default: false)* If enabled, the model will be allowed to train on  an unjoined dataset, where some examples might be false negative and might appear in the dataset later as (true) positive example.


## Constructors

### Operator


```
SigmoidCrossEntropyWithLogitsOp(
    logits *core.Tensor,
    targets *core.Tensor,
    out *core.Tensor,
    logDTrick bool,
    unjoinedLrLoss bool) core.Operator

SigmoidCrossEntropyWithLogitsGradientOp(
    gradOut *core.Tensor,
    logits *core.Tensor,
    targets *core.Tensor,
    gradLogits *core.Tensor,
    logDTrick bool,
    unjoinedLrLoss bool) core.Operator
```


### Layer


```
func(f *Fragment) SigmoidCrossEntropyWithLogits(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func SigmoidCrossEntropyWithLogits(
        g api.Graph, 
        t core.Type, 
        shape []int, 
        logDTrick bool, 
        unjoinedLrLoss bool) {
    logits := g.NewTensor(t)
    targets := g.NewTensor(t)
    out := g.NewTensor(t)
    g.External(logits, shape)
    g.External(targets, shape)
    g.SigmoidCrossEntropyWithLogitsOp(logits, targets, out, logDTrick, unjoinedLrLoss)
    g.Return(out)
}
```


### Layer


```
import "fragata/arhat/front/models"

func SigmoidCrossEntropyWithLogits(
        m *models.Model, 
        t string, 
        shape []int, 
        logDTrick bool, 
        unjoinedLrLoss bool) {
    logits := m.External("dtype", t, "shape", shape)
    targets := m.External("dtype", t, "shape", shape)
    out := 
        m.SigmoidCrossEntropyWithLogits(
            logits, 
            targets, 
            "logDTrick", logDTrick, 
            "unjoinedLrLoss", unjoinedLrLoss)
    m.Return(out, 0)
}
```

