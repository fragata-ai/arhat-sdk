
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
        kernelSize []int,
        stride []int,
        padding []int) front.Layer

type InceptionBlocks struct {
    ConvBlock ConvBlockFunc
    InceptionA func(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inChannels int, 
        poolFeatures int, 
        convBlock ConvBlockFunc) front.Layer
    InceptionB func(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inChannels int, 
        convBlock ConvBlockFunc) front.Layer
    InceptionC func(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inChannels int,
        channels7x7 int,
        convBlock ConvBlockFunc) front.Layer
    InceptionD func(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inChannels int,
        convBlock ConvBlockFunc) front.Layer
    InceptionE func(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inChannels int,
        convBlock ConvBlockFunc) front.Layer
    InceptionAux func(
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

type Inception3Args struct {
    numClasses int
    training bool
    auxLogits bool
    transformInput bool
    inceptionBlocks *InceptionBlocks
}

// construction/destruction

func NewInception3Args() *Inception3Args {
    return &Inception3Args{
        numClasses: 1000,
        training: false,
        auxLogits: true,
        transformInput: false,
        inceptionBlocks: nil,
    }
}

// interface

func(a *Inception3Args) SetNumClasses(v int) { a.numClasses = v }
func(a *Inception3Args) SetTraining(v bool) { a.training = v }
func(a *Inception3Args) SetAuxLogits(v bool) { a.auxLogits = v }
func(a *Inception3Args) SetTransformInput(v bool) { a.transformInput = v }
func(a *Inception3Args) SetInceptionBlocks(v *InceptionBlocks) { a.inceptionBlocks = v }

//
//    Model builder
//

// Inception v3 model architecture from
// "Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`
//
// **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
//     N x 3 x 299 x 299, so ensure your images are sized accordingly. 

func BuildInception3(batchSize int, args *Inception3Args) *models.Model {
    if args == nil {
        args = NewInception3Args()
    }
    r := 
        NewInception3( 
            args.numClasses, 
            args.training, 
            args.auxLogits, 
            args.transformInput,
            args.inceptionBlocks)
    m := models.NewModel()
    x := m.External("shape", []int{batchSize, 3, 299, 299})
    // aux is reserved for future use
    y, _ := r.Forward(m, x)
    y = m.Softmax(y, "axis", 1)
    m.Return(y, 0)
    return m
}

//
//    Inception3
//

type Inception3 struct {
    numClasses int
    training bool
    auxLogits bool
    transformInput bool
    inceptionBlocks *InceptionBlocks
}

// construction/destruction

func NewInception3(
        numClasses int,
        training bool,
        auxLogits bool,
        transformInput bool,
        inceptionBlocks *InceptionBlocks) *Inception3 {
    p := new(Inception3)
    p.numClasses = numClasses
    p.training = training
    p.auxLogits = auxLogits
    p.transformInput = transformInput
    if inceptionBlocks == nil {
        inceptionBlocks = 
            &InceptionBlocks{
                BasicConv2d,
                InceptionA,
                InceptionB,
                InceptionC,
                InceptionD,
                InceptionE,
                InceptionAux,
            }
    }
    p.inceptionBlocks = inceptionBlocks
    return p
}

// interface

func(p *Inception3) Forward(m *models.Model, x front.Layer) (front.Layer, front.Layer) {
    // SKIPPED: TransformInput
    b := p.inceptionBlocks
    convBlock := b.ConvBlock
    inceptionA := b.InceptionA
    inceptionB := b.InceptionB
    inceptionC := b.InceptionC
    inceptionD := b.InceptionD
    inceptionE := b.InceptionE
    inceptionAux := b.InceptionAux
    // N x 3 x 299 x 299
    x = convBlock(m, "Conv2d_1a_3x3", x, 3, 32, D(3), D(2), nil)
    // N x 32 x 149 x 149
    x = convBlock(m, "Conv2d_2a_3x3", x, 32, 32, D(3), nil, nil)
    // N x 32 x 147 x 147
    x = convBlock(m, "Conv2d_2b_3x3", x, 32, 64, D(3), nil, D(1))
    // N x 64 x 147 x 147
    x = maxPool2d(m, x, 3, 2, 0)
    // N x 64 x 73 x 73
    x = convBlock(m, "Conv2d_3b_1x1", x, 64, 80, D(1), nil, nil)
    // N x 80 x 73 x 73
    x = convBlock(m, "Conv2d_4a_3x3", x, 80, 192, D(3), nil, nil)
    // N x 192 x 71 x 71
    x = maxPool2d(m, x, 3, 2, 0)
    // N x 192 x 35 x 35
    x = inceptionA(m, "Mixed_5b", x, 192, 32, nil)
    // N x 256 x 35 x 35
    x = inceptionA(m, "Mixed_5c", x, 256, 64, nil)
    // N x 288 x 35 x 35
    x = inceptionA(m, "Mixed_5d", x, 288, 64, nil)
    // N x 288 x 35 x 35
    x = inceptionB(m, "Mixed_6a", x, 288, nil)
    // N x 768 x 17 x 17
    x = inceptionC(m, "Mixed_6b", x, 768, 128, nil)
    // N x 768 x 17 x 17
    x = inceptionC(m, "Mixed_6c", x, 768, 160, nil)
    // N x 768 x 17 x 17
    x = inceptionC(m, "Mixed_6d", x, 768, 160, nil)
    // N x 768 x 17 x 17
    x = inceptionC(m, "Mixed_6e", x, 768, 192, nil)
    // N x 768 x 17 x 17
    var aux front.Layer
    if p.training && p.auxLogits {
        aux = inceptionAux(m, "AuxLogits", x, 768, p.numClasses, nil)
    }
    // N x 768 x 17 x 17
    x = inceptionD(m, "Mixed_7a", x, 768, nil)
    // N x 1280 x 8 x 8
    x = inceptionE(m, "Mixed_7b", x, 1280, nil)
    // N x 2048 x 8 x 8
    x = inceptionE(m, "Mixed_7c", x, 2048, nil)
    // Adaptive average pooling
    x = m.AveragePool(x, "globalPooling", true, "kernel", []int{1, 1})
    // N x 2048 x 1 x 1
    x = m.Dropout(x)
    // N x 2048 x 1 x 1
    x = linear(m, "fc", x, 2048, p.numClasses)
    // N x 1000 (numClasses)
    return x, aux
}

func InceptionA(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inChannels int, 
        poolFeatures int, 
        convBlock ConvBlockFunc) front.Layer {
    if convBlock == nil {
        convBlock = BasicConv2d
    }
    branch1x1 := convBlock(m, id+"_branch1x1", x, inChannels, 64, D(1), nil, nil)
    branch5x5 := convBlock(m, id+"_branch5x5_1", x, inChannels, 48, D(1), nil, nil)
    branch5x5 = convBlock(m, id+"_branch5x5_2", branch5x5, 48, 64, D(5), nil, D(2))
    branch3x3dbl := convBlock(m, id+"_branch3x3dbl_1", x, inChannels, 64, D(1), nil, nil)
    branch3x3dbl = convBlock(m, id+"_branch3x3dbl_2", branch3x3dbl, 64, 96, D(3), nil, D(1))
    branch3x3dbl = convBlock(m, id+"_branch3x3dbl_3", branch3x3dbl, 96, 96, D(3), nil, D(1))
    branchPool := avgPool2d(m, x, 3, 1, 1)
    branchPool = convBlock(m, id+"_branch_pool", branchPool, inChannels, poolFeatures, D(1), nil, nil) 
    return m.Concat(branch1x1, branch5x5, branch3x3dbl, branchPool, "axis", 1)   
}

func InceptionB(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inChannels int, 
        convBlock ConvBlockFunc) front.Layer {
    if convBlock == nil {
        convBlock = BasicConv2d
    }
    branch3x3 := convBlock(m, id+"_branch3x3", x, inChannels, 384, D(3), D(2), nil)
    branch3x3dbl := convBlock(m, id+"_branch3x3dbl_1", x, inChannels, 64, D(1), nil, nil)
    branch3x3dbl = convBlock(m, id+"_branch3x3dbl_2", branch3x3dbl, 64, 96, D(3), nil, D(1))
    branch3x3dbl = convBlock(m, id+"_branch3x3dbl_3", branch3x3dbl, 96, 96, D(3), D(2), nil)
    branchPool := maxPool2d(m, x, 3, 2, 0)
    return m.Concat(branch3x3, branch3x3dbl, branchPool, "axis", 1)
}

func InceptionC(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inChannels int,
        channels7x7 int,
        convBlock ConvBlockFunc) front.Layer {
    if convBlock == nil {
        convBlock = BasicConv2d
    }
    c7 := channels7x7
    branch1x1 := convBlock(m, id+"_branch1x1", x, inChannels, 192, D(1), nil, nil)
    branch7x7 := convBlock(m, id+"_branch7x7_1", x, inChannels, c7, D(1), nil, nil)
    branch7x7 = convBlock(m, id+"_branch7x7_2", branch7x7, c7, c7, HW(1, 7), nil, HW(0, 3))
    branch7x7 = convBlock(m, id+"_branch7x7_3", branch7x7, c7, 192, HW(7, 1), nil, HW(3, 0))
    branch7x7dbl := convBlock(m, id+"_branch7x7dbl_1", x, inChannels, c7, D(1), nil, nil)
    branch7x7dbl = convBlock(m, id+"_branch7x7dbl_2", branch7x7dbl, c7, c7, HW(7, 1), nil, HW(3, 0))
    branch7x7dbl = convBlock(m, id+"_branch7x7dbl_3", branch7x7dbl, c7, c7, HW(1, 7), nil, HW(0, 3))
    branch7x7dbl = convBlock(m, id+"_branch7x7dbl_4", branch7x7dbl, c7, c7, HW(7, 1), nil, HW(3, 0))
    branch7x7dbl = convBlock(m, id+"_branch7x7dbl_5", branch7x7dbl, c7, 192, HW(1, 7), nil, HW(0, 3))
    branchPool := avgPool2d(m, x, 3, 1, 1)
    branchPool = convBlock(m, id+"_branch_pool", branchPool, inChannels, 192, D(1), nil, nil)
    return m.Concat(branch1x1, branch7x7, branch7x7dbl, branchPool, "axis", 1)
}

func InceptionD(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inChannels int,
        convBlock ConvBlockFunc) front.Layer {
    if convBlock == nil {
        convBlock = BasicConv2d
    }
    branch3x3 := convBlock(m, id+"_branch3x3_1", x, inChannels, 192, D(1), nil, nil)
    branch3x3 = convBlock(m, id+"_branch3x3_2", branch3x3, 192, 320, D(3), D(2), nil)
    branch7x7x3 := convBlock(m, id+"_branch7x7x3_1", x, inChannels, 192, D(1), nil, nil)
    branch7x7x3 = convBlock(m, id+"_branch7x7x3_2", branch7x7x3, 192, 192, HW(1, 7), nil, HW(0, 3))
    branch7x7x3 = convBlock(m, id+"_branch7x7x3_3", branch7x7x3, 192, 192, HW(7, 1), nil, HW(3, 0))
    branch7x7x3 = convBlock(m, id+"_branch7x7x3_4", branch7x7x3, 192, 192, D(3), D(2), nil)
    branchPool := maxPool2d(m, x, 3, 2, 0)
    return m.Concat(branch3x3, branch7x7x3, branchPool, "axis", 1)
}

func InceptionE(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inChannels int,
        convBlock ConvBlockFunc) front.Layer {
    if convBlock == nil {
        convBlock = BasicConv2d
    }
    branch1x1 := convBlock(m, id+"_branch1x1", x, inChannels, 320, D(1), nil, nil)
    branch3x3 := convBlock(m, id+"_branch3x3_1", x, inChannels, 384, D(1), nil, nil)
    branch3x3A := convBlock(m, id+"_branch3x3_2a", branch3x3, 384, 384, HW(1, 3), nil, HW(0, 1))
    branch3x3B := convBlock(m, id+"_branch3x3_2b", branch3x3, 384, 384, HW(3, 1), nil, HW(1, 0))
    branch3x3 = m.Concat(branch3x3A, branch3x3B, "axis", 1)
    branch3x3dbl := convBlock(m, id+"_branch3x3dbl_1", x, inChannels, 448, D(1), nil, nil)
    branch3x3dbl = convBlock(m, id+"_branch3x3dbl_2", branch3x3dbl, 448, 384, D(3), nil, D(1))
    branch3x3dblA := convBlock(m, id+"_branch3x3dbl_3a", branch3x3dbl, 384, 384, HW(1, 3), nil, HW(0, 1))
    branch3x3dblB := convBlock(m, id+"_branch3x3dbl_3b", branch3x3dbl, 384, 384, HW(3, 1), nil, HW(1, 0))
    branch3x3dbl = m.Concat(branch3x3dblA, branch3x3dblB, "axis", 1)
    branchPool := avgPool2d(m, x, 3, 1, 1)
    branchPool = convBlock(m, id+"_branch_pool", branchPool, inChannels, 192, D(1), nil, nil) 
    return m.Concat(branch1x1, branch3x3, branch3x3dbl, branchPool, "axis", 1)
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
    // SKIPPED: conv1.stddev, fc.stddev (originally used by initializers)
    // N x 768 x 17 x 17
    x = avgPool2d(m, x, 5, 3, 0)
    // N x 768 x 5 x 5
    x = convBlock(m, id+"_conv0", x, inChannels, 128, D(1), nil, nil)
    // N x 128 x 5 x 5
    x = convBlock(m, id+"_conv1", x, 128, 786, D(5), nil, nil)
    // N x 768 x 1 x 1
    // Adaptive average pooling
    x = m.AveragePool(x, "globalPooling", true, "kernel", []int{1, 1})
    // N x 768 x 1 x 1
    x = linear(m, id+"_fc", x, 768, numClasses)
    // N x 1000
    return x
}

func BasicConv2d(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inChannels int,
        outChannels int,
        kernelSize []int,
        stride []int,
        padding []int) front.Layer {
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
        kernelSize []int, 
        stride []int, 
        padding []int) front.Layer {
    if padding != nil {
        padding = []int{padding[0], padding[1], padding[0], padding[1]}
    }
    w := 
        m.Variable(
            "label", id+"_weight", 
            "shape", []int{outChannels, inChannels, kernelSize[0], kernelSize[1]})
    return m.Conv(
        x,
        w,
        nil,
        "kernel", kernelSize,
        "stride", stride,
        "pads", padding)
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

func D(n int) []int {
    return []int{n, n}
}

func HW(h int, w int) []int {
    return []int{h, w}
}

