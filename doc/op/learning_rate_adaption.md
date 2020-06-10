
# LearningRateAdaption

Learning Rate Adaption is an operation that perform one iteration of
gradient descent based on learning rate:

    lr(k) = lr(k-1) - lrAlpha * df(k-1)/dlr,

where `df(k-1)/dlr` is the gradient of objective function `f` on `lr`, and
`lrAlpha` is a learning rate hyperparameter. It can be prove that
`df(k-1)/dlr` equals `INNERPRODUCT(grad(k-1), -grad(k-2))`, where `grad(k-1)` is
the grad of `f(k-1)` on parameters. When the argument
`normalizedLrAdaption` is *false*, we simply perform the
following update:

    lr(k) = lr(k-1) - lrAlpha * INNERPRODUCT(grad(k-1), grad(k-2)).

If we set `normalizedLrAdaption` to be *true*, we do not directly apply
`INNERPRODUCT(grad(k-1), -grad(k-2))` as the grad. Instead, we perform the
following update:

    lr(k) = lr(k-1) + lrAlpha * cosineSimilarity(grad(k-1), grad(k-2)).

## Interface

### Input

**lr**

>Learning rate

**grad**

>Gradient computed

**effgrad**

>The effective grad

### Output

**outputLr**

>Updated learning rate

### Attributes

**lrAlpha**

>the learning rate for performing gradient descent on learning rate `lr`

**normalizedLrAdaption**

>*(type: bool; default: false)* whether to apply normalized lr adaption or not


## Constructors

### Operator


```
LearningRateAdaptionOp(
    lr *core.Tensor,
    grad *core.Tensor,
    effgrad *core.Tensor,
    outputLr *core.Tensor,
    lrAlpha float32,
    normalizedLrAdaption bool) core.Operator
```

