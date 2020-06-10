
# Ftlr

Implements Follow the Regularised Leader (FTRL) algorithm.

## Constructors

### Operator


```
FtrlOp(
    param *core.Tensor,
    nz *core.Tensor,
    grad *core.Tensor,
    outputParam *core.Tensor,
    outputNz *core.Tensor,
    alpha float32,
    beta float32,
    lambda1 float32,
    lambda2 float32) core.Operator
```

