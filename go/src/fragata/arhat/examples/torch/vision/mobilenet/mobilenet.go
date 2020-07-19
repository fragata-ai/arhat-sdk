
package main

import (
    "fmt"
    "math"
    front "fragata/arhat/front/core"
    "fragata/arhat/front/models"
)

//
//    Local types
//

type NormLayerFunc func(m *models.Model, id string, x front.Layer, planes int) front.Layer 
type BlockFunc func(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inp int, 
        oup int, 
        stride int, 
        expandRatio float32,
        normLayer NormLayerFunc) front.Layer

//
//    Optional arguments
//

type MobileNetV2Args struct {
    numClasses int
    widthMult float32
    invertedResidualSetting [][4]int
    roundNearest int
    block BlockFunc
    normLayer NormLayerFunc
}

// construction/destruction

func NewMobileNetV2Args() *MobileNetV2Args {
    return &MobileNetV2Args{
        numClasses: 1000,
        widthMult: 1.0,
        invertedResidualSetting: nil,
        roundNearest: 8,
        block: nil,
        normLayer: nil,
    }
}

// interface

func(a *MobileNetV2Args) SetNumClasses(v int) { a.numClasses = v }
func(a *MobileNetV2Args) SetWidthMult(v float32) { a.widthMult = v }
func(a *MobileNetV2Args) SetInvertedResidualSetting(v [][4]int) { a.invertedResidualSetting = v }
func(a *MobileNetV2Args) SetRoundNearest(v int) { a.roundNearest = v }
func(a *MobileNetV2Args) SetBlock(v BlockFunc) { a.block = v }
func(a *MobileNetV2Args) SetNIrmLayer(v NormLayerFunc) { a.normLayer = v }

//
//    Model builder
//

// Constructs a MobileNetV2 architecture from
// "MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>

func BuildMobileNetV2(batchSize int, args *MobileNetV2Args) *models.Model {
    if args == nil {
        args = NewMobileNetV2Args()
    }
    r := 
        NewMobileNetV2( 
            args.numClasses, 
            args.widthMult, 
            args.invertedResidualSetting, 
            args.roundNearest,
            args.block,
            args.normLayer)
    m := models.NewModel()
    x := m.External("shape", []int{batchSize, 3, 224, 224})
    y := r.Forward(m, x)
    y = m.Softmax(y, "axis", 1)
    m.Return(y, 0)
    return m
}

//
//    MobileNetV2
//

type MobileNetV2 struct {
    numClasses int
    widthMult float32
    invertedResidualSetting [][4]int
    roundNearest int
    block BlockFunc
    normLayer NormLayerFunc
}

// construction/destruction

var defaultInvertedResidualSetting = [][4]int{
    // t, c, n, s
    [4]int{1, 16, 1, 1},
    [4]int{6, 24, 2, 2},
    [4]int{6, 32, 3, 2},
    [4]int{6, 64, 4, 2},
    [4]int{6, 96, 3, 1},
    [4]int{6, 160, 3, 2},
    [4]int{6, 320, 1, 1},
}

func NewMobileNetV2(
        numClasses int,
        widthMult float32,
        invertedResidualSetting [][4]int,
        roundNearest int,
        block BlockFunc,
        normLayer NormLayerFunc) *MobileNetV2 {
    if invertedResidualSetting == nil {
        invertedResidualSetting = defaultInvertedResidualSetting
    }
    if block == nil {
        block = InvertedResidual
    }
    if normLayer == nil {
        normLayer = batchNorm2d
    }
    p := new(MobileNetV2)
    p.numClasses = numClasses
    p.widthMult = widthMult
    p.invertedResidualSetting = invertedResidualSetting
    p.roundNearest = roundNearest
    p.block = block
    p.normLayer = normLayer
    return p
}

// interface

func(p *MobileNetV2) Forward(m *models.Model, x front.Layer) front.Layer {
    widthMult := p.widthMult
    roundNearest := p.roundNearest
    block := p.block
    normLayer := p.normLayer
    inputChannel := 32
    lastChannel := 1280
    // building first layer
    inputChannel = makeDivisible(float32(inputChannel)*widthMult, roundNearest, -1)
    v := float32(lastChannel)
    if widthMult > 1.0 {
        v *= widthMult
    }
    lastChannel = makeDivisible(v, roundNearest, -1)
    suffix := 0
    makeId := func() string {
        s := fmt.Sprintf("features_%d", suffix)
        suffix++
        return s
    }
    x = ConvBnRelu(m, makeId(), x, 3, inputChannel, 3, 2, 1, normLayer)
    // building inverted residual blocks
    for _, dims := range p.invertedResidualSetting {
        t := dims[0]
        c := dims[1]
        n := dims[2]
        s := dims[3]
        outputChannel := makeDivisible(float32(c)*widthMult, roundNearest, -1)
        for i := 0; i < n; i++ {
            stride := 1
            if i == 0 {
                stride = s
            }
            x = block(m, makeId(), x, inputChannel, outputChannel, stride, float32(t), normLayer)
            inputChannel = outputChannel
        }
    }
    // building last several layers
    x = ConvBnRelu(m, makeId(), x, inputChannel, lastChannel, 1, 1, 1, normLayer)
    x = m.AveragePool(x, "globalPooling", true, "kernel", []int{1, 1})
    // building classifier
    x = m.Dropout(x, "ratio", 0.2)
    x = linear(m, "classifier_1", x, lastChannel, p.numClasses)
    return x
}

// implementation

func InvertedResidual(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inp int, 
        oup int, 
        stride int, 
        expandRatio float32,
        normLayer NormLayerFunc) front.Layer {
    if normLayer == nil {
        normLayer = batchNorm2d
    }
    hiddenDim := int(math.Round(float64(inp)*float64(expandRatio)))
    useResConnect := (stride == 1 && inp == oup)
    suffix := 0
    makeId := func() string {
        s := fmt.Sprintf("%s_conv_%d", id, suffix)
        suffix++
        return s
    }
    y := x
    if expandRatio != 1.0 {
        // pw
        y = ConvBnRelu(m, makeId(), y, inp, hiddenDim, 1, 1, 1, normLayer) 
    }
    // dw
    y = ConvBnRelu(m, makeId(), y, hiddenDim, hiddenDim, 3, stride, hiddenDim, normLayer)
    // pw-linear
    y = conv2d(m, makeId(), y, hiddenDim, oup, 1, 1, 0, 1)
    y = normLayer(m, makeId(), y, oup)
    if useResConnect {
        y = m.Add(x, y)
    }
    return y
}

func ConvBnRelu(
        m *models.Model,
        id string,
        x front.Layer,
        inPlanes int,
        outPlanes int,
        kernelSize int,
        stride int,
        groups int,
        normLayer NormLayerFunc) front.Layer {
    padding := (kernelSize - 1) / 2
    if normLayer == nil {
        normLayer = batchNorm2d
    }
    x = conv2d(m, id+"_0", x, inPlanes, outPlanes, kernelSize, stride, padding, groups)
    x = normLayer(m, id+"_1", x, outPlanes)
    x = m.ReluN(x, "n", 6.0)
    return x
}

func makeDivisible(v float32, divisor int, minValue int) int {
    // This function is taken from the original tf repo.
    // It ensures that all layers have a channel number that is divisible by 8
    // It can be seen here:
    // https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    if minValue < 0 {
        minValue = divisor
    }
    newV := (int(v+float32(divisor)/2.0) / divisor) * divisor
    if newV < minValue {
        newV = minValue
    }
    // Make sure that round down does not go down by more than 10%.
    if float32(newV) < 0.9 * v {
        newV += divisor
    }
    return newV
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

func batchNorm2d(m *models.Model, id string, x front.Layer, channels int) front.Layer {
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
    return m.SpatialBn(x, scale, bias, runningMean, runningVar, "epsilon", 1.0e-5)
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

