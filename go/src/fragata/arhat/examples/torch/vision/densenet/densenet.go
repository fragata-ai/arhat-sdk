
package main

import (
    "fmt"
    front "fragata/arhat/front/core"
    "fragata/arhat/front/models"
)

//
//    Optional arguments
//

type DenseNetArgs struct {
    bnSize int
    dropRate float32
    numClasses int
    memoryEfficient bool
}

func NewDenseNetArgs() *DenseNetArgs {
    return &DenseNetArgs{
        bnSize: 4,
        dropRate: 0.0,
        numClasses: 1000,
        memoryEfficient: false,
    }
}

func(a *DenseNetArgs) SetBnSize(v int) { a.bnSize = v }
func(a *DenseNetArgs) SetDropRate(v float32) { a.dropRate = v }
func(a *DenseNetArgs) SetNumClasses(v int) { a.numClasses = v }
func(a *DenseNetArgs) SetMemoryEfficient(v bool) { a.memoryEfficient = v }

//
//    Model builder
//

func BuildDenseNet(
        batchSize int,
        growthRate int,
        blockConfig []int,
        numInitFeatures int,
        args *DenseNetArgs) *models.Model {
    if args == nil {
        args = NewDenseNetArgs()
    }
    r := 
        NewDenseNet(
            growthRate,
            blockConfig,
            numInitFeatures,
            args.bnSize,
            args.dropRate,
            args.numClasses,
            args.memoryEfficient)
    m := models.NewModel()
    x := m.External("shape", []int{batchSize, 3, 224, 224})
    y := r.Forward(m, x)
    y = m.Softmax(y, "axis", 1)
    m.Return(y, 0)
    return m
}

//
//    DenseNet
//

type DenseNet struct {
    growthRate int
    blockConfig []int
    numInitFeatures int
    bnSize int
    dropRate float32
    numClasses int
    memoryEfficient bool
}

// construction/destruction

func NewDenseNet(
        growthRate int,
        blockConfig []int,
        numInitFeatures int,
        bnSize int,
        dropRate float32,
        numClasses int,
        memoryEfficient bool) *DenseNet {
    p := new(DenseNet)
    p.growthRate = growthRate
    p.blockConfig = blockConfig
    p.numInitFeatures = numInitFeatures
    p.bnSize = bnSize
    p.dropRate = dropRate
    p.numClasses = numClasses
    p.memoryEfficient = memoryEfficient
    return p
}

// interface

func(p *DenseNet) Forward(m *models.Model, x front.Layer) front.Layer {
    // features
    x = conv2d(m, "features_conv0", x, 3, p.numInitFeatures, 7, 2, 3)
    x = batchNorm2d(m, "features_norm0", x, p.numInitFeatures)
    x = m.Relu(x)
    x = 
        m.MaxPool(
            x, 
            "kernel", []int{3, 3}, 
            "stride", []int{2, 2}, 
            "pads", []int{1, 1, 1, 1})
    // each denseblock
    numFeatures := p.numInitFeatures
    for i, numLayers := range p.blockConfig {
        x = 
            DenseBlock(
                m,
                fmt.Sprintf("features_denseblock%d", i+1),
                x,
                numLayers,
                numFeatures,
                p.bnSize,
                p.growthRate,
                p.dropRate,
                p.memoryEfficient)
        numFeatures += numLayers * p.growthRate
        if i != len(p.blockConfig) - 1 {
            x = 
                Transition(
                    m, 
                    fmt.Sprintf("features_transition%d", i+1), 
                    x, 
                    numFeatures, 
                    numFeatures/2)
            numFeatures /= 2
        }
    }
    // Final batch norm
    x = batchNorm2d(m, "features_norm5", x, numFeatures) 
    x = m.Relu(x)
    x = m.AveragePool(x, "globalPooling", true, "kernel", []int{1, 1})
    // classifier
    x = linear(m, "classifier", x, numFeatures, p.numClasses)
    return x
}

func DenseBlock(
        m *models.Model,
        id string,
        x front.Layer,
        numLayers int,
        numInputFeatures int,
        bnSize int,
        growthRate int,
        dropRate float32,
        memoryEfficient bool) front.Layer {
    features := []front.Layer{x}
    for i := 0; i < numLayers; i++ {
        layer := 
            DenseLayer(
                m,
                fmt.Sprintf("%s_denselayer%d", id, i+1),
                features,
                numInputFeatures+i*growthRate,
                growthRate,
                bnSize,
                dropRate,
                memoryEfficient)
        features = append(features, layer)
    }
    return concat(m, features)
}

func DenseLayer(
        m *models.Model,
        id string,
        input []front.Layer,
        numInputFeatures int,
        growthRate int,
        bnSize int,
        dropRate float32,
        memoryEfficient bool) front.Layer {
    // memoryEfficient is not supported
    x := concat(m, input)
    x = batchNorm2d(m, id+"_norm_1", x, numInputFeatures) 
    x = m.Relu(x)
    x = conv2d(m, id+"_conv_1", x, numInputFeatures, bnSize*growthRate, 1, 1, 0)
    x = batchNorm2d(m, id+"_norm_2", x, bnSize*growthRate)
    x = m.Relu(x)
    x = conv2d(m, id+"_conv_2", x, bnSize*growthRate, growthRate, 3, 1, 1)
    if dropRate > 0.0 {
        x = m.Dropout(x, "ratio", dropRate)
    }
    return x
}

func Transition(
        m *models.Model,
        id string,
        x front.Layer,
        numInputFeatures int,
        numOutputFeatures int) front.Layer {
    x = batchNorm2d(m, id+"_norm", x, numInputFeatures)
    x = m.Relu(x)
    x = conv2d(m, id+"_conv", x, numInputFeatures, numOutputFeatures, 1, 1, 0)
    x = m.AveragePool(x, "kernel", []int{2, 2}, "stride", []int{2, 2})
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
    return m.SpatialBn(x, scale, bias, runningMean, runningVar)
}

func linear(m *models.Model, id string, x front.Layer, inChannels int, outChannels int) front.Layer {
    w := 
        m.Variable(
            "label", id+"_weight", 
            "shape", []int{outChannels, inChannels, 1, 1})
    b := 
        m.Variable(
            "label", id+"_bias",
            "shape", []int{outChannels})
    return m.FullyConnected(x, w, b)
}

func concat(m *models.Model, input []front.Layer) front.Layer {
    count := len(input)
    args := make([]interface{}, count+2)
    for i := 0; i < count; i++ {
        args[i] = input[i]
    }
    args[count] = "axis"
    args[count+1] = 1
    return m.Concat(args...)
}

