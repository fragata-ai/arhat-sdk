
package main

import (
    front "fragata/arhat/front/core"
    "fragata/arhat/front/models"
)

//
//    Local types
//

type ConvBlockFunc func(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inChannels int,
        outChannels int,
        kernelSize int,
        stride int,
        padding int) front.Layer

type InceptionBlocks struct {
    ConvBlock ConvBlockFunc
    InceptionBlock func(
        m *models.Model,
        id string,
        x front.Layer,
        inChannels int,
        ch1x1 int,
        ch3x3red int,
        ch3x3 int,
        ch5x5red int,
        ch5x5 int,
        poolProj int,
        convBlock ConvBlockFunc) front.Layer
    InceptionAuxBlock func(
        m *models.Model,
        id string,
        x front.Layer,
        inChannels int,
        numClasses int,
        convBlock ConvBlockFunc) front.Layer
}

//
//    Optional arguments
//

type GoogLeNetArgs struct {
    numClasses int
    training bool
    auxLogits bool
    transformInput bool
    inceptionBlocks *InceptionBlocks
}

// construction/destruction

func NewGoogLeNetArgs() *GoogLeNetArgs {
    return &GoogLeNetArgs{
        numClasses: 1000,
        training: false,
        auxLogits: true,
        transformInput: false,
        inceptionBlocks: nil,
    }
}

// interface

func(a *GoogLeNetArgs) SetNumClasses(v int) { a.numClasses = v }
func(a *GoogLeNetArgs) SetTraining(v bool) { a.training = v }
func(a *GoogLeNetArgs) SetAuxLogits(v bool) { a.auxLogits = v }
func(a *GoogLeNetArgs) SetTransformInput(v bool) { a.transformInput = v }
func(a *GoogLeNetArgs) SetInceptionBlocks(v *InceptionBlocks) { a.inceptionBlocks = v }

//
//    Model builder
//

// GoogLeNet (Inception v1) model architecture from
// "Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842> 

func BuildGoogLeNet(batchSize int, args *GoogLeNetArgs) *models.Model {
    if args == nil {
        args = NewGoogLeNetArgs()
    }
    r := 
        NewGoogLeNet( 
            args.numClasses, 
            args.training, 
            args.auxLogits, 
            args.transformInput,
            args.inceptionBlocks)
    m := models.NewModel()
    x := m.External("shape", []int{batchSize, 3, 224, 224})
    // aux1, aux2 are reserved for future use
    y, _, _ := r.Forward(m, x)
    y = m.Softmax(y, "axis", 1)
    m.Return(y, 0)
    return m
}

//
//    GoogLeNet
//

type GoogLeNet struct {
    numClasses int
    training bool
    auxLogits bool
    transformInput bool
    inceptionBlocks *InceptionBlocks
}

// construction/destruction

func NewGoogLeNet(
        numClasses int,
        training bool,
        auxLogits bool,
        transformInput bool,
        inceptionBlocks *InceptionBlocks) *GoogLeNet {
    p := new(GoogLeNet)
    p.numClasses = numClasses
    p.training = training
    p.auxLogits = auxLogits
    p.transformInput = transformInput
    if inceptionBlocks == nil {
        inceptionBlocks = 
            &InceptionBlocks{
                BasicConv2d,
                Inception,
                InceptionAux,
            }
    }
    p.inceptionBlocks = inceptionBlocks
    return p
}

// interface

func(p *GoogLeNet) Forward(m *models.Model, x front.Layer) (front.Layer, front.Layer, front.Layer) {
    // SKIPPED: TransformInput
    b := p.inceptionBlocks
    convBlock := b.ConvBlock
    inceptionBlock := b.InceptionBlock
    inceptionAuxBlock := b.InceptionAuxBlock
    // N x 3 x 224 x 224
    x = convBlock(m, "conv1", x, 3, 64, 7, 2, 3)
    // N x 64 x 112 x 112
    x = maxPool2d(m, x, 3, 2, 0)
    // N x 64 x 56 x 56
    x = convBlock(m, "conv2", x, 64, 64, 1, 1, 0)
    // N x 64 x 56 x 56
    x = convBlock(m, "conv3", x, 64, 192, 3, 1, 1)
    // N x 192 x 56 x 56
    x = maxPool2d(m, x, 3, 2, 0)
    // N x 192 x 28 x 28
    x = inceptionBlock(m, "inception3a", x, 192, 64, 96, 128, 16, 32, 32, nil)
    // N x 256 x 28 x 28
    x = inceptionBlock(m, "inception3b", x, 256, 128, 128, 192, 32, 96, 64, nil)
    // N x 480 x 28 x 28
    x = maxPool2d(m, x, 3, 2, 0)
    // N x 480 x 14 x 14
    x = inceptionBlock(m, "inception4a", x, 480, 192, 96, 208, 16, 48, 64, nil)
    // N x 512 x 14 x 14
    var aux1 front.Layer
    if p.training && p.auxLogits {
        aux1 = inceptionAuxBlock(m, "aux1", x, 512, p.numClasses, nil)
    }
    x = inceptionBlock(m, "inception4b", x, 512, 160, 112, 224, 24, 64, 64, nil)
    // N x 512 x 14 x 14
    x = inceptionBlock(m, "inception4c", x, 512, 128, 128, 256, 24, 64, 64, nil)
    // N x 512 x 14 x 14
    x = inceptionBlock(m, "inception4d", x, 512, 112, 144, 288, 32, 64, 64, nil)
    // N x 528 x 14 x 14
    var aux2 front.Layer
    if p.training && p.auxLogits {
        aux2 = inceptionAuxBlock(m, "aux2", x, 528, p.numClasses, nil)
    }
    x = inceptionBlock(m, "inception4e", x, 528, 256, 160, 320, 32, 128, 128, nil)
    // N x 832 x 14 x 14
    x = maxPool2d(m, x, 2, 2, 0)
    // N x 832 x 7 x 7
    x = inceptionBlock(m, "inception5a", x, 832, 256, 160, 320, 32, 128, 128, nil)
    // N x 832 x 7 x 7
    x = inceptionBlock(m, "inception5b", x, 832, 384, 192, 384, 48, 128, 128, nil)
    // N x 1024 x 7 x 7
    x = m.AveragePool(x, "globalPooling", true, "kernel", []int{1, 1})
    // N x 1024 x 1 x 1
    x = m.Dropout(x)
    x = linear(m, "fc", x, []int{p.numClasses, 1024, 1, 1})
    // N x 1000 (numClasses)
    return x, aux2, aux1
}

func Inception(
        m *models.Model,
        id string,
        x front.Layer,
        inChannels int,
        ch1x1 int,
        ch3x3red int,
        ch3x3 int,
        ch5x5red int,
        ch5x5 int,
        poolProj int,
        convBlock ConvBlockFunc) front.Layer {
    if convBlock == nil {
        convBlock = BasicConv2d
    }
    branch1 := convBlock(m, id+"_branch1", x, inChannels, ch1x1, 1, 1, 0)
    branch2 := convBlock(m, id+"_branch2_0", x, inChannels, ch3x3red, 1, 1, 0)
    branch2 = convBlock(m, id+"_branch2_1", branch2, ch3x3red, ch3x3, 3, 1, 1)
    branch3 := convBlock(m, id+"_branch3_0", x, inChannels, ch5x5red, 1, 1, 0)
    // Here, kernelSize=3 instead of kernelSize=5 is a known bug.
    // Please see https://github.com/pytorch/vision/issues/906 for details.
    branch3 = convBlock(m, id+"_branch3_1", branch3, ch5x5red, ch5x5, 3, 1, 1)
    branch4 := maxPool2d(m, x, 3, 1, 1)
    branch4 = convBlock(m, id+"_branch4_1", branch4, inChannels, poolProj, 1, 1, 0)
    return m.Concat(branch1, branch2, branch3, branch4, "axis", 1)
}

func InceptionAux(
        m *models.Model,
        id string,
        x front.Layer,
        inChannels int,
        numClasses int,
        convBlock ConvBlockFunc) front.Layer {
    if convBlock == nil {
        convBlock = BasicConv2d
    }
    // aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
    // PyTorch adaptive_avg_pool is replaced with regular average pooling from the original paper
    x = avgPool2d(m, x, 5, 3, 0)
    // aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
    x = convBlock(m, id+"_conv", x, inChannels, 128, 1, 1, 0)
    // N x 128 x 4 x 4
    x = linear(m, id+"_fc1", x, []int{1024, 128, 4, 4})
    x = m.Relu(x)
    // N x 1024
    x = m.Dropout(x, "ratio", 0.7)
    // N x 1024
    x = linear(m, id+"_fc2", x, []int{numClasses, 1024})
    // N x 1000 (numClasses)
    return x
}

func BasicConv2d(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inChannels int, 
        outChannels int,
        kernelSize int,
        stride int,
        padding int) front.Layer {
    x = conv2d(m, id+"_conv", x, inChannels, outChannels, kernelSize, stride, padding)
    x = batchNorm2d(m, id+"_bn", x, outChannels)
    x = m.Relu(x)
    return x
}
        
func conv2d(
        m *models.Model, 
        id string, 
        x front.Layer,
        inChannels int, 
        outChannels int, 
        kernelSize int, 
        stride int, 
        padding int) front.Layer {
    w := 
        m.Variable(
            "label", id+"_weight", 
            "shape", []int{outChannels, inChannels, kernelSize, kernelSize})
    return m.Conv(
        x,
        w,
        nil,
        "kernel", []int{kernelSize, kernelSize},
        "stride", []int{stride, stride},
        "pads", []int{padding, padding, padding, padding})
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
    return m.SpatialBn(x, scale, bias, runningMean, runningVar, "epsilon", 0.001)
}

func avgPool2d(
        m *models.Model, 
        x front.Layer, 
        kernelSize int, 
        stride int, 
        padding int) front.Layer {
    return m.AveragePool(
        x, 
        "kernel", []int{kernelSize, kernelSize}, 
        "stride", []int{stride, stride},
        "pads", []int{padding, padding, padding, padding})
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

func linear(m *models.Model, id string, x front.Layer, weightShape []int) front.Layer {
    w := 
        m.Variable(
            "label", id+"_weight", 
            "shape", weightShape)
    b := 
        m.Variable(
            "label", id+"_bias",
            "shape", []int{weightShape[0]})
    return m.FullyConnected(x, w, b)
}

