
package main

import (
    "fmt"
    front "fragata/arhat/front/core"
    "fragata/arhat/front/models"
)

//
//    Local types
//

type InvertedResidualFunc func(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inp int, 
        oup int, 
        stride int) front.Layer

//
//    Optional arguments
//

type ShuffleNetV2Args struct {
    numClasses int
    invertedResidual InvertedResidualFunc
}

// construction/destruction

func NewShuffleNetV2Args() *ShuffleNetV2Args {
    return &ShuffleNetV2Args{
        numClasses: 1000, 
        invertedResidual: InvertedResidual,
    }
}

// inteface

func(a *ShuffleNetV2Args) SetNumClasses(v int) { a.numClasses = v }
func(a *ShuffleNetV2Args) SetInvertedResidualFunc(v InvertedResidualFunc) { a.invertedResidual = v }

//
//    Model builder
//

func BuildShuffleNetV2(
        batchSize int,
        stagesRepeats []int,
        stagesOutChannels []int,
        args *ShuffleNetV2Args) *models.Model {
    if args == nil {
        args = NewShuffleNetV2Args()
    }
    r := NewShuffleNetV2(stagesRepeats, stagesOutChannels, args.numClasses, args.invertedResidual)
    m := models.NewModel()
    x := m.External("shape", []int{batchSize, 3, 224, 224})
    y := r.Forward(m, x)
    y = m.Softmax(y, "axis", 1)
    m.Return(y, 0)
    return m
}

//
//    ShuffleNetV2
//

type ShuffleNetV2 struct {
    stagesRepeats []int
    stagesOutChannels []int
    numClasses int
    invertedResidual InvertedResidualFunc
}

// construction/destruction

func NewShuffleNetV2(
        stagesRepeats []int,
        stagesOutChannels []int,
        numClasses int,
        invertedResidual InvertedResidualFunc) *ShuffleNetV2 {
    p := new(ShuffleNetV2)
    p.Init(stagesRepeats, stagesOutChannels, numClasses, invertedResidual)
    return p
}

func(p *ShuffleNetV2) Init(
        stagesRepeats []int,
        stagesOutChannels []int,
        numClasses int,
        invertedResidual InvertedResidualFunc) {
    p.stagesRepeats = stagesRepeats
    p.stagesOutChannels = stagesOutChannels
    p.numClasses = numClasses
    p.invertedResidual = invertedResidual
}

// interface

func(p *ShuffleNetV2) Forward(m *models.Model, x front.Layer) front.Layer {
    stagesRepeats := p.stagesRepeats
    if len(stagesRepeats) != 3 {
        fatal(front.ValueError("expected stagesRepeats as list of 3 positive ints"))
    }
    stagesOutChannels := p.stagesOutChannels
    if len(stagesOutChannels) != 5 {
        fatal(front.ValueError("expected stagesOutChannels as list of 5 positive ints"))
    }
    invertedResidual := p.invertedResidual
    inputChannels := 3
    outputChannels := stagesOutChannels[0]
    x = conv2d(m, "conv1_0", x, inputChannels, outputChannels, 3, 2, 1, 1)
    x = batchNorm2d(m, "conv1_1", x, outputChannels)
    x = m.Relu(x)
    inputChannels = outputChannels
    x = maxPool2d(m, x, 3, 2, 1)
    for i := 0; i < 3; i++ {
        name := fmt.Sprintf("stage%d", i+2)
        repeats := stagesRepeats[i]
        outputChannels := stagesOutChannels[i+1]
        x = invertedResidual(m, name+"_0", x, inputChannels, outputChannels, 2)
        for k := 1; k < repeats; k++ {
            id := fmt.Sprintf("%s_%d", name, k)
            x = invertedResidual(m, id, x, outputChannels, outputChannels, 1)
        }
        inputChannels = outputChannels
    }
    outputChannels = stagesOutChannels[4]
    x = conv2d(m, "conv5_0", x, inputChannels, outputChannels, 1, 1, 0, 1)
    x = batchNorm2d(m, "conv5_1", x, outputChannels)
    x = m.Relu(x)
    x = m.AveragePool(x, "globalPooling", true, "kernel", []int{1, 1})
    x = linear(m, "fc", x, outputChannels, p.numClasses)
    return x
}

// implementation

func InvertedResidual(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inp int, 
        oup int, 
        stride int) front.Layer {
    if !(stride >= 1 && stride <= 3) {
        fatal(front.ValueError("illegal stride value"))
    }
    branchFeatures := oup / 2
    var x1, x2 interface{}
    var features int
    if stride == 1 {
        if !(branchFeatures * 2 == inp) {
            fatal(front.AssertionError("Mismatched inp and oup for stride=1"))
        }
        chunks := m.Split(x, "axis", 1, "split", []int{branchFeatures, branchFeatures})
        x1 = chunks.OutputAt(0)
        x2 = chunks.OutputAt(1)
        features = branchFeatures
    } else {
        branch1 := id + "_branch1"
        x1 = conv2d(m, branch1+"_0", x, inp, inp, 3, stride, 1, inp)
        x1 = batchNorm2d(m, branch1+"_1", x1, inp)
        x1 = conv2d(m, branch1+"_2", x1, inp, branchFeatures, 1, 1, 0, 1)
        x1 = batchNorm2d(m, branch1+"_3", x1, branchFeatures)
        x1 = m.Relu(x1)
        x2 = x
        features = inp
    }
    branch2 := id + "_branch2"
    x2 = conv2d(m, branch2+"_0", x2, features, branchFeatures, 1, 1, 0, 1)
    x2 = batchNorm2d(m, branch2+"_1", x2, branchFeatures)
    x2 = m.Relu(x2)
    x2 = conv2d(m, branch2+"_3", x2, branchFeatures, branchFeatures, 3, stride, 1, branchFeatures)
    x2 = batchNorm2d(m, branch2+"_4", x2, branchFeatures)
    x2 = conv2d(m, branch2+"_5", x2, branchFeatures, branchFeatures, 1, 1, 0, 1)
    x2 = batchNorm2d(m, branch2+"_6", x2, branchFeatures)
    x2 = m.Relu(x2)
    out := m.Concat(x1, x2, "axis", 1)
    out = channelShuffle(m, out, 2)
    return out
}

func channelShuffle(m *models.Model, x front.Layer, groups int) front.Layer {
    return m.ChannelShuffle(x, "group", groups)
}

func conv2d(
        m *models.Model, 
        id string, 
        x interface{},
        inChannels int, 
        outChannels int, 
        kernelSize int, 
        stride int, 
        padding int,
        groups int) front.Layer {
    validateInput(x)
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
        x interface{}, 
        channels int) front.Layer {
    validateInput(x)
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
    return m.SpatialBn(x, scale, bias, runningMean, runningVar)
}

func maxPool2d(
        m *models.Model, 
        x front.Layer, 
        kernelSize int, 
        stride int, 
        padding int) front.Layer {
    return m.MaxPool(
        x, 
        "kernel", []int{kernelSize, kernelSize}, 
        "stride", []int{stride, stride},
        "pads", []int{padding, padding, padding, padding})
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

func validateInput(x interface{}) {
    switch x.(type) {
    case front.Layer, *front.Data:
        // ok
    default:
        fatal(front.AssertionError("Invalid input type: %T", x))
    }
}

func fatal(err error) {
    front.Throw(err)
}

