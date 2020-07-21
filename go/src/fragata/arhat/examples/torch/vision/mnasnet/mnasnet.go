
package main

import (
    "fmt"
    front "fragata/arhat/front/core"
    "fragata/arhat/front/models"
)

//
//    Optional arguments
//

type MnasNetArgs struct {
    numClasses int
    dropout float32
}

// construction/destruction

func NewMnasNetArgs() *MnasNetArgs {
    return &MnasNetArgs{
        numClasses: 1000,
        dropout: 0.2,
    }
}

// interface

func(a *MnasNetArgs) SetNumClasses(v int) { a.numClasses = v }
func(a *MnasNetArgs) SetDropout(v float32) { a.dropout = v }

//
//    Model builder
//

func BuildMnasNet(batchSize int, alpha float32, args *MnasNetArgs) *models.Model {
    if args == nil {
        args = NewMnasNetArgs()
    }
    r := NewMnasNet(alpha, args.numClasses, args.dropout) 
    m := models.NewModel()
    x := m.External("shape", []int{batchSize, 3, 224, 224})
    y := r.Forward(m, x)
    y = m.Softmax(y, "axis", 1)
    m.Return(y, 0)
    return m
}

//
//    MnasNet
//

type MnasNet struct {
    alpha float32
    numClasses int
    dropout float32
    depths []int
    bnMomentum float32
}

// construction/destruction

func NewMnasNet(alpha float32, numClasses int, dropout float32) *MnasNet {
    p := new(MnasNet)
    p.Init(alpha, numClasses, dropout)
    return p
}

func(p *MnasNet) Init(alpha float32, numClasses int, dropout float32) {
    p.alpha = alpha
    p.numClasses = numClasses
    p.dropout = dropout
    p.depths = getDepths(alpha)
    // Paper suggests 0.9997 momentum
    p.bnMomentum = 0.9997
}

// interface

func(p *MnasNet) Forward(m *models.Model, x front.Layer) front.Layer {
    depths := p.depths
    bnMomentum := p.bnMomentum
    // First layer: regular conv.
    x = conv2d(m, "layers_0", x, 3, depths[0], 3, 2, 1, 1)
    x = batchNorm2d(m, "layers_1", x, depths[0], bnMomentum)
    x = m.Relu(x)
    // Depthwise separable, no skip.
    x = conv2d(m, "layers_3", x, depths[0], depths[0], 3, 1, 1, depths[0])
    x = batchNorm2d(m, "layers_4", x, depths[0], bnMomentum)
    x = m.Relu(x)
    x = conv2d(m, "layers_6", x, depths[0], depths[1], 1, 1, 0, 1)
    x = batchNorm2d(m, "layers_7", x, depths[1], bnMomentum)
    // MNASNet blocks: stacks of inverted residuals.
    x = Stack(m, "layers_8", x, depths[1], depths[2], 3, 2, 3, 3, bnMomentum)
    x = Stack(m, "layers_9", x, depths[2], depths[3], 5, 2, 3, 3, bnMomentum)
    x = Stack(m, "layers_10", x, depths[3], depths[4], 5, 2, 6, 3, bnMomentum)
    x = Stack(m, "layers_11", x, depths[4], depths[5], 3, 1, 6, 2, bnMomentum)
    x = Stack(m, "layers_12", x, depths[5], depths[6], 5, 2, 6, 4, bnMomentum)
    x = Stack(m, "layers_13", x, depths[6], depths[7], 3, 1, 6, 1, bnMomentum)
    // Final mapping to classifier input.
    x = conv2d(m, "layers_14", x, depths[7], 1280, 1, 1, 0, 1)
    x = batchNorm2d(m, "layers_15", x, 1280, bnMomentum)
    x = m.Relu(x)
    // Global avgpool to remove H and W dimensions.
    x = m.AveragePool(x, "globalPooling", true, "kernel", []int{1, 1})
    // classifier
    x = m.Dropout(x, "ratio", p.dropout)
    x = linear(m, "classifier_1", x, 1280, p.numClasses)
    return x
}

// implementation

func Stack(
        m *models.Model,
        id string,
        x front.Layer,
        inCh int,
        outCh int,
        kernelSize int,
        stride int,
        expFactor int,
        repeats int,
        bnMomentum float32) front.Layer {
    suffix := 0
    makeId := func() string {
        s := fmt.Sprintf("%s_%d", id, suffix)
        suffix++
        return s
    }
    // First one has no skip, because feature map size changes.
    x = InvertedResidual(m, makeId(), x, inCh, outCh, kernelSize, stride, expFactor, bnMomentum)
    for i := 1; i < repeats; i++ {
        x = InvertedResidual(m, makeId(), x, outCh, outCh, kernelSize, 1, expFactor, bnMomentum)
    }
    return x
}

func InvertedResidual(
        m *models.Model,
        id string,
        x front.Layer, 
        inCh int,
        outCh int,
        kernelSize int,
        stride int,
        expansionFactor int,
        bnMomentum float32) front.Layer {
    midCh := inCh * expansionFactor
    applyResidual := (inCh == outCh && stride == 1)
    id += "_layers"
    y := x
    // Pointwise
    x = conv2d(m, id+"_0", x, inCh, midCh, 1, 1, 0, 1)
    x = batchNorm2d(m, id+"_1", x, midCh, bnMomentum)
    x = m.Relu(x)
    // Depthwise
    x = conv2d(m, id+"_3", x, midCh, midCh, kernelSize, stride, kernelSize/2, midCh)
    x = batchNorm2d(m, id+"_4", x, midCh, bnMomentum)
    x = m.Relu(x)
    // Linear pointwise. Note that there is no activation.
    x = conv2d(m, id +"_6", x, midCh, outCh, 1, 1, 0, 1)
    x = batchNorm2d(m, id+"_7", x, outCh, bnMomentum)
    if applyResidual {
        x = m.Add(x, y)
    }
    return x
}

func getDepths(alpha float32) []int {
    // Scales tensor depths as in reference MobileNet code,
    // prefers rouding up rather than down.
    depths := []int{32, 16, 24, 40, 80, 96, 192, 320}
    count := len(depths)
    for i := 0; i < count; i++ {
        depths[i] = roundToMultipleOf(float32(depths[i])*alpha, 8, -1.0)
    }
    return depths
}

func roundToMultipleOf(val float32, divisor int, roundUpBias float32) int {
    // Asymmetric rounding to make `val` divisible by `divisor`. With default
    // bias, will round up, unless the number is no more than 10% greater than the
    // smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88.
    if roundUpBias < 0.0 {
        roundUpBias = 0.9
    }
    newVal := (int(val+float32(divisor)/2.0) / divisor) * divisor
    if newVal < divisor {
        newVal = divisor
    }
    if float32(newVal) < roundUpBias * val {
        newVal += divisor
    }
    return newVal
}

func conv2d(
        m *models.Model, 
        id string, 
        x front.Layer,
        inChannels int, 
        outChannels int, 
        kernelSize int, 
        stride int, 
        padding int,
        groups int) front.Layer {
    w := 
        m.Variable(
            "label", id+"_weight", 
            "shape", []int{outChannels, inChannels/groups, kernelSize, kernelSize})
    return m.Conv(
        x, 
        w,
        nil,
        "kernel", []int{kernelSize, kernelSize},
        "stride", []int{stride, stride},
        "pads", []int{padding, padding, padding, padding},
        "group", groups)
}

func batchNorm2d(
        m *models.Model, 
        id string, 
        x front.Layer, 
        channels int, 
        momentum float32) front.Layer {
    scale :=
        m.Variable(
            "label", id+"_weight", 
            "shape", []int{channels})
    bias := 
        m.Variable(
            "label", id+"_bias", 
            "shape", []int{channels})
    runningMean := 
        m.Variable(
            "label", id+"_running_mean", 
            "shape", []int{channels})
    runningVar := 
        m.Variable(
            "label", id+"_running_var", 
            "shape", []int{channels})
    return m.SpatialBn(x, scale, bias, runningMean, runningVar, "momentum", momentum)
}

func linear(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inPlanes int, 
        outPlanes int) front.Layer {
    w := 
        m.Variable(
            "label", id+"_weight", 
            "shape", []int{outPlanes, inPlanes, 1, 1})
    b := 
        m.Variable(
            "label", id+"_bias",
            "shape", []int{outPlanes})
    return m.FullyConnected(x, w, b)
}

