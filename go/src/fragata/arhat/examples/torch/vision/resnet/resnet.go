
package main

import (
    "fmt"
    "fragata/arhat/core"
    front "fragata/arhat/front/core"
    "fragata/arhat/front/models"
)

//
//    Local types
//

type DownsampleFunc func(m *models.Model, x front.Layer) front.Layer 
type NormFunc func(m *models.Model, x front.Layer, planes int) front.Layer 

type Block interface {
    Expansion() int
    Forward(
        m *models.Model,
        x front.Layer,
        inPlanes int,
        planes int,
        stride int,
        downsample DownsampleFunc,
        groups int,
        baseWidth int,
        dilation int,
        normLayer NormFunc) front.Layer
}

//
//    Optional arguments
//

//
//    TODO: Add pointer to "front/initializers".Initializer as optional argument;
//        if it is not nil, construct and add initiaizers as you go
//

type ResNetArgs struct {
    numClasses int
    zeroInitResidual bool
    groups int
    widthPerGroup int
    replaceStrideWithDilation []bool
    normLayer NormFunc
    // TODO: Add object to hold initializers here
}

func NewResNetArgs() *ResNetArgs {
    return &ResNetArgs{
        numClasses: 1000,
        zeroInitResidual: false,
        groups: 1,
        widthPerGroup: 64,
        replaceStrideWithDilation: nil,
        normLayer: nil,
    }
}

func(a *ResNetArgs) SetNumClasses(v int) { a.numClasses = v }
func(a *ResNetArgs) SetZeroInitResidual(v bool) { a.zeroInitResidual = v }
func(a *ResNetArgs) SetGroups(v int) { a.groups = v }
func(a *ResNetArgs) SetWidthPerGroup(v int) { a.widthPerGroup = v }
func(a *ResNetArgs) SetReplaceStrideWithDilation(v []bool) { a.replaceStrideWithDilation = v }
func(a *ResNetArgs) SetNormLayer(v NormFunc) { a.normLayer = v }

//
//    Model builder
//

func BuildResNet(batchSize int, block Block, layers []int, args *ResNetArgs) *models.Model {
    resetLabelIndex()
    if args == nil {
        args = NewResNetArgs()
    }
    r := 
        NewResNet(
            block, 
            layers, 
            args.numClasses, 
            args.zeroInitResidual, 
            args.groups, 
            args.widthPerGroup,
            args.replaceStrideWithDilation,
            args.normLayer)
    m := models.NewModel()
    x := m.External("shape", []int{batchSize, 3, 224, 224})
    y := r.Forward(m, x)
    y = m.Softmax(y, "axis", 1)
    m.Return(y, 0)
    return m
}

//
//    ResNet
//

type ResNet struct {
    normLayer NormFunc
    inPlanes int
    dilation int
    replaceStrideWithDilation []bool
    groups int
    baseWidth int
    block Block
    layers []int
    numClasses int
    zeroInitResidual bool
}

// construction/destruction

func NewResNet(
        block Block,
        layers []int,
        numClasses int,
        zeroInitResidual bool,
        groups int,
        widthPerGroup int,
        replaceStrideWithDilation []bool,
        normLayer NormFunc) *ResNet {
    p := new(ResNet)
    if normLayer == nil {
        normLayer = batchNorm
    }
    p.normLayer = normLayer
    p.inPlanes = 64
    p.dilation = 1
    // each element in the slice indicates if we should replace
    // the 2x2 stride with a dilated convolution instead
    n := len(replaceStrideWithDilation) 
    if n == 0 {
        p.replaceStrideWithDilation = []bool{false, false, false}
    } else {
        if n != 3 {
            fatal(front.ValueError(
                "replaceStrideWithDilation should be nil or a 3-element slice, got %d", n))
        }
        p.replaceStrideWithDilation = make([]bool, n) 
        copy(p.replaceStrideWithDilation, replaceStrideWithDilation)
    }
    p.groups = groups
    p.baseWidth = widthPerGroup
    p.block = block
    if len(layers) != 4 {
        fatal(front.ValueError(
            "layers should be a 4-element slice, got %d", len(layers)))
    }
    p.layers = core.IntsClone(layers)
    p.numClasses = numClasses
    p.zeroInitResidual = zeroInitResidual
    return p
}

// interface

func(p *ResNet) Forward(m *models.Model, x front.Layer) front.Layer {
    conv1 := conv7x7(m, x, 3, p.inPlanes, 2, 3)
    bn1 := p.normLayer(m, conv1, p.inPlanes)
    relu1 := m.Relu(bn1)
    maxpool := 
        m.MaxPool(
            relu1, 
            "kernel", []int{3, 3}, 
            "stride", []int{2, 2}, 
            "pads", []int{1, 1, 1, 1})
    layer1 := p.makeLayer(m, maxpool, p.block, 64, p.layers[0], 1, false)
    layer2 := p.makeLayer(m, layer1, p.block, 128, p.layers[1], 2, p.replaceStrideWithDilation[0])
    layer3 := p.makeLayer(m, layer2, p.block, 256, p.layers[2], 2, p.replaceStrideWithDilation[1])
    layer4 := p.makeLayer(m, layer3, p.block, 512, p.layers[3], 2, p.replaceStrideWithDilation[2])
    // originally nn.AdaptiveAvgPool2d
    avgpool := m.ReduceMean(layer4, "axes", []int{2, 3})
    n := 512 * p.block.Expansion()
    reshape := m.Reshape(avgpool, "shape", []int{-1, n})
    out := fullyConnected(m, reshape, n, p.numClasses)
    return out
}

// implementation

func(p *ResNet) makeLayer(
        m *models.Model,
        x front.Layer,
        block Block,
        planes int,
        blocks int,
        stride int,
        dilate bool) front.Layer {
    normLayer := p.normLayer
    var downsample DownsampleFunc
    previousDilation := p.dilation
    if dilate {
        p.dilation *= stride
        stride = 1
    }
    outPlanes := planes * block.Expansion()
    if stride != 1 || p.inPlanes != outPlanes {
        downsample = func(m *models.Model, x front.Layer) front.Layer {
            conv := conv1x1(m, x, p.inPlanes, outPlanes, stride)
            return normLayer(m, conv, outPlanes)
        }
    }
    y := 
        block.Forward(
            m, 
            x,
            p.inPlanes,
            planes,
            stride,
            downsample,
            p.groups,
            p.baseWidth,
            previousDilation,
            normLayer)
    p.inPlanes = outPlanes
    for i := 1; i < blocks; i++ { 
        y = 
            block.Forward(
                m,
                y,
                p.inPlanes,
                planes,
                1,   // stride
                nil, // downsample
                p.groups,
                p.baseWidth,
                p.dilation,
                normLayer)
    }
    return y
}

//
//    BasicBlock
//

type BasicBlock struct {}

// construction/destruction

func NewBasicBlock() *BasicBlock {
    return new(BasicBlock)
}

// interface

func(b *BasicBlock) Expansion() int {
    return 1
}

func(b *BasicBlock) Forward(
        m *models.Model,
        x front.Layer,
        inPlanes int,
        planes int,
        stride int,
        downsample DownsampleFunc,
        groups int,
        baseWidth int,
        dilation int,
        normLayer NormFunc) front.Layer {
    if groups != 1 || baseWidth != 64 {
        fatal(front.ValueError("BasicBlock only supports groups=1 and baseWidth=64"))
    }
    if dilation > 1 {
        fatal(front.NotImplemented("Dilation > 1 not supported in BasicBlock"))
    }
    if normLayer == nil {
        normLayer = batchNorm
    }
    // Both conv1 and downsample layers downsample the input when stride != 1
    conv1 := conv3x3(m, x, inPlanes, planes, stride, 1, 1)
    bn1 := normLayer(m, conv1, planes)
    relu1 := m.Relu(bn1)
    conv2 := conv3x3(m, relu1, planes, planes, 1, 1, 1)
    bn2 := normLayer(m, conv2, planes)
    identity := x
    if downsample != nil {
        identity = downsample(m, x)
    }
    add := m.Add(bn2, identity)
    out := m.Relu(add)
    return out
}

//
//    Bottleneck
//

type Bottleneck struct { }

// construction/destruction

func NewBottleneck() *Bottleneck {
    return new(Bottleneck)
}

// interface

func(b *Bottleneck) Expansion() int {
    return 4
}

func(b *Bottleneck) Forward(
        m *models.Model,
        x front.Layer,
        inPlanes int,
        planes int,
        stride int,
        downsample DownsampleFunc,
        groups int,
        baseWidth int,
        dilation int,
        normLayer NormFunc) front.Layer {
    // Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    // while original implementation places the stride at the first 1x1 convolution(self.conv1)
    // according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    // This variant is also known as ResNet V1.5 and improves accuracy according to
    // https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    expansion := b.Expansion()
    if normLayer == nil {
        normLayer = batchNorm
    }
    width := int(float32(planes) * (float32(baseWidth) / 64.0)) * groups
    // Both conv2 and downsample layers downsample the input when stride != 1
    conv1 := conv1x1(m, x, inPlanes, width, 1)
    bn1 := normLayer(m, conv1, width)
    relu1 := m.Relu(bn1)
    conv2 := conv3x3(m, relu1, width, width, stride, groups, dilation)
    bn2 := normLayer(m, conv2, width)
    relu2 := m.Relu(bn2)
    conv3 := conv1x1(m, relu2, width, planes*expansion, 1)
    bn3 := normLayer(m, conv3, planes*expansion)
    identity := x
    if downsample != nil {
        identity = downsample(m, x)
    }
    add := m.Add(bn3, identity)
    out := m.Relu(add)
    return out
}

//
//    Local functions
//

func batchNorm(m *models.Model, x front.Layer, planes int) front.Layer {
    labelIndex := makeLabelIndex()
    scale :=
        m.Variable(
            "label", fmt.Sprintf("bn_weight_%d", labelIndex), 
            "shape", []int{1, planes})
    bias := 
        m.Variable(
            "label", fmt.Sprintf("bn_bias_%d", labelIndex), 
            "shape", []int{1, planes})
    estMean := 
        m.Variable(
            "label", fmt.Sprintf("bn_mean_%d", labelIndex), 
            "shape", []int{1, planes})
    estVar := 
        m.Variable(
            "label", fmt.Sprintf("bn_var_%d", labelIndex), 
            "shape", []int{1, planes})
    // TODO: Remove "isTest" after redesign of SpatialBn
    return m.SpatialBn(x, scale, bias, estMean, estVar, "isTest", true)
}

func conv7x7(
        m *models.Model,
        x front.Layer,
        inPlanes int, 
        outPlanes int, 
        stride int,
        padding int) front.Layer {
    w, b := makeConvVars(m, inPlanes, outPlanes, 7, false)
    return m.Conv(
        x, 
        w, 
        b, 
        "kernel", []int{7, 7}, 
        "stride", []int{stride, stride},
        "pads", []int{padding, padding, padding, padding})
}

func conv3x3(
        m *models.Model, 
        x front.Layer, 
        inPlanes int, 
        outPlanes int,
        stride int,
        groups int,
        dilation int) front.Layer {
    // 3x3 convolution with padding
    w, b := makeConvVars(m, inPlanes, outPlanes, 3, false)
    return m.Conv(
        x, 
        w, 
        b, 
        "kernel", []int{3, 3}, 
        "dilation", []int{dilation, dilation},
        "stride", []int{stride, stride},
        "pads", []int{dilation, dilation, dilation, dilation},
        "group", groups)
}

func conv1x1(
        m *models.Model, 
        x front.Layer, 
        inPlanes int, 
        outPlanes int,
        stride int) front.Layer {
    // 1x1 convolution
    w, b := makeConvVars(m, inPlanes, outPlanes, 1, false)
    return m.Conv(
        x, 
        w, 
        b, 
        "kernel", []int{1, 1}, 
        "stride", []int{stride, stride})
}

func fullyConnected(m *models.Model, x front.Layer, inPlanes int, outPlanes int) front.Layer {
    labelIndex := makeLabelIndex()
    w := 
        m.Variable(
            "label", fmt.Sprintf("fc_weight_%d", labelIndex), 
            "shape", []int{outPlanes, inPlanes})
    b := 
        m.Variable(
            "label", fmt.Sprintf("fc_bias_%d", labelIndex),
            "shape", []int{outPlanes})
    return m.FullyConnected(x, w, b)
}

func makeConvVars(
        m *models.Model, inPlanes int, outPlanes int, kernel int, bias bool) (
            w front.Layer, b front.Layer) {
    labelIndex := makeLabelIndex()
    w = 
        m.Variable(
            "label", fmt.Sprintf("conv_weight_%d", labelIndex), 
            "shape", []int{outPlanes, inPlanes, kernel, kernel})
    if bias {
        b = 
            m.Variable(
                "label", fmt.Sprintf("conv_bias_%d", labelIndex),
                "shape", []int{outPlanes})
    }
    return
}

var nextLabelIndex = 1

func resetLabelIndex() {
    nextLabelIndex = 1
}

func makeLabelIndex() int {
    index := nextLabelIndex
    nextLabelIndex++
    return index
}

func fatal(err error) {
    front.Throw(err)
}

