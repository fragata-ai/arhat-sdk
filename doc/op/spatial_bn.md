
# SpatialBn

Applies spatial batch normalization to the input tensor as described in the original paper, 
[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). 
Be aware, this operator has two different output sets, depending on the value of *isTest*. 
According to the paper, the primary operation of spatial batch normalization is:

    y = (x - mu_x) / SQRT(sigma_x^2 + epsilon) * gamma + b

In the equation, `mu_x` is the *mean*, `x` is the input data, `sigma_x^2` is the *var*, 
`epsilon` is *epsilon*, `gamma` is the *scale*, `b` is the *bias*, 
and `y` is the output data. The *momentum* arg also affects this calculation in 
the computation of the running mean and variance. The influence of *momentum* is as follows:

    runningMean = runningMean * momentum + mean * (1 - momentum)

    runningVar = runningVar * momentum + var * (1 - momentum)

Output when `isTest` is *false* (train mode): *y, runningMean, runningVar, savedMean, savedVar*

Output when `isTest` is *true* (test mode): *y* (all other output tensors are set to *nil*)

Note that although in training mode *runningMean* and *runningVar* are updated 
at each call based on their previous values and are, strictly speaking, input-output
arguments, for simplicty they are treated as pure output arguments and the framework
implementation will take care of their proper initialization and updating.

Supported types: *float, half*.

## Interface

### Input

**x**

>The input 4-dimensional tensor of shape `(N, C, H, W)`

**scale**

>The scale as a 1-dimensional tensor of size `C` to be applied to the output

**bias**

>The bias as a 1-dimensional tensor of size `C` to be applied to the output

**estMean**

>The estimated mean as a 1-dimensional tensor of size `C`.  Used for testing only, should be *nil* for training.


**estVar**

>The estimated variance as a 1-dimensional tensor of size `C`. Used for testing only, should be *nil* for training.


### Output

**y**

>The output 4-dimensional tensor of the same shape as `x`

**savedMean**

>Saved mean used during training to speed up gradient computation.  Should not be used for testing.


**savedInvStd**

>Saved inverted variance used during training to speed up gradient computation.  Should not be used for testing.


**runningMean**

>The running mean after the spatial BN operator.  Used for training only, should be *nil* for testing.


**runningVar**

>The running variance after the spatial BN operator.  Used for training only, should be *nil* for testing.


### Attributes

**isTest**

>*(type: bool; default: false)* If set to *true*,  run spatial batch normalization in test mode; otherwise run in train mode


**epsilon**

>*(type: float; default: 1e-5)* The epsilon value to use to avoid division by zero


**momentum**

>*(type: float; default: 0.9)* Factor used in computing the running mean and variance


**numBatches**

>*(type: int; default: 1)* Specifies the number of batches to apply normalization on.  Currently not supported.


## Constructors

### Operator


```
SpatialBnOp(
    x *core.Tensor,
    scale *core.Tensor,
    bias *core.Tensor,
    estMean *core.Tensor,
    estVar *core.Tensor,
    y *core.Tensor,
    savedMean *core.Tensor,
    savedInvStd *core.Tensor,
    runningMean *core.Tensor,
    runningVar *core.Tensor,
    isTest bool,
    epsilon float64,
    momentum float32,
    numBatches int) core.Operator

SpatialBnGradientOp(
    x *core.Tensor,
    scale *core.Tensor,
    dy *core.Tensor,
    savedMean *core.Tensor,
    savedRstd *core.Tensor,
    dx *core.Tensor,
    dscale *core.Tensor,
    dbias *core.Tensor,
    epsilon float64,
    numBatches int) core.Operator
```


### Layer


```
func(f *Fragment) SpatialBn(args ...interface{}) front.Layer
```


## Examples

### Operator


```
import (
    "fragata/arhat/core"
    "fragata/arhat/graph/api"
)

func SpatialBnTrain(
        g api.Graph, 
        t core.Type, 
        shape []int, 
        epsilon float64, 
        momentum float32, 
        numBatches int) {
    x := g.NewTensor(t)
    scale := g.NewTensor(t)
    bias := g.NewTensor(t)
    y := g.NewTensor(t)
    savedMean := g.NewTensor(t)
    savedInvStd := g.NewTensor(t)
    runningMean := g.NewTensor(t)
    runningVar := g.NewTensor(t)
    c := shape[1:2]
    g.External(x, shape)
    g.External(scale, c)
    g.External(bias, c) 
    g.SpatialBnOp(
        x, 
        scale, 
        bias, 
        nil, // estMean
        nil, // estVar
        y,
        savedMean,
        savedInvStd,
        runningMean,
        runningVar,
        false, 
        epsilon, 
        momentum,
        numBatches)
    g.Return(y)
    g.Return(savedMean)
    g.Return(savedInvStd)
    g.Return(runningMean)
    g.Return(runningVar)
}

func SpatialBnTest(g api.Graph, t core.Type, shape []int, epsilon float64, numBatches int) {
    x := g.NewTensor(t)
    scale := g.NewTensor(t)
    bias := g.NewTensor(t)
    estMean := g.NewTensor(t)
    estVar := g.NewTensor(t)
    y := g.NewTensor(t)
    c := shape[1:2]
    g.External(x, shape)
    g.External(scale, c)
    g.External(bias, c)
    g.External(estMean, c)
    g.External(estVar, c) 
    g.SpatialBnOp(
        x, 
        scale, 
        bias, 
        estMean,
        estVar,
        y,
        nil, // savedMean
        nil, // savedInvStd
        nil, // runningMean
        nil, // runningVar
        true, 
        epsilon,
        0.0, // momentum
        numBatches)
    g.Return(y)
}
```


### Layer


```
import "fragata/arhat/front/models"

func SpatialBnTrain(
        m *models.Model, 
        t string, 
        shape []int, 
        epsilon float64, 
        momentum float32, 
        numBatches int) {
    c := shape[1:2]
    x := m.External("dtype", t, "shape", shape)
    scale := m.External("dtype", t, "shape", c)
    bias := m.External("dtype", t, "shape", c)
    bn := 
        m.SpatialBn(
            x, 
            scale, 
            bias, 
            nil, // estMean
            nil, // estVar
            "isTest", false, 
            "epsilon", epsilon, 
            "momentum", momentum,
            "numBatches", numBatches)
    m.Return(bn, 0) // y
    m.Return(bn, 1) // savedMean
    m.Return(bn, 2) // savedInvStd
    m.Return(bn, 3) // runningMean
    m.Return(bn, 4) // runningVar
}

func SpatialBnTest(m *models.Model, t string, shape []int, epsilon float64, numBatches int) {
    c := shape[1:2]
    x := m.External("dtype", t, "shape", shape)
    scale := m.External("dtype", t, "shape", c)
    bias := m.External("dtype", t, "shape", c)
    estMean := m.External("dtype", t, "shape", c)
    estVar := m.External("dtype", t, "shape", c)
    bn := 
        m.SpatialBn(
            x, 
            scale, 
            bias, 
            estMean,
            estVar,
            "isTest", true, 
            "epsilon", epsilon, 
            "numBatches", numBatches)
    m.Return(bn, 0) // y
}
```

