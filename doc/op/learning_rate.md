
# LearningRate

Learning rate is a decreasing function of time. With low learning rates the
improvements will be linear. With high learning rates they will start to look
more exponential. Learning rate can be controlled by the various policies.

Supported types: *float*.

## Interface

### Output

**output**

>Computed learning rate

### Attributes

**baseLr**

>*(type: float; required)*

**policy**

>learning rate policy name and attributes

## Constructors

### Operator


```
LearningRateOp(
    output *core.Tensor, 
    baseLr float32, 
    policy sgd.LearningRatePolicy) core.Operator
```

