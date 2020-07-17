
package main

import (
    "fmt"
    front "fragata/arhat/front/core"
    "fragata/arhat/front/models"
)

//
//    Model builder
//

func BuildSqueezeNet(batchSize int, version string, numClasses int) *models.Model {
    r := NewSqueezeNet(version, numClasses)
    m := models.NewModel()
    x := m.External("shape", []int{batchSize, 3, 224, 224})
    y := r.Forward(m, x)
    y = m.Softmax(y, "axis", 1)
    m.Return(y, 0)
    return m
}

//
//    SqueezeNet
//

type SqueezeNet struct {
    version string
    numClasses int
}

// construction/destruction

func NewSqueezeNet(version string, numClasses int) *SqueezeNet {
    p := new(SqueezeNet)
    p.version = version
    p.numClasses = numClasses
    return p
}

// interface

func(p *SqueezeNet) Forward(m *models.Model, x front.Layer) front.Layer {
    // features
    switch p.version {
    case "1_0":
        x = conv2d(m, "features_0", x, 3, 96, 7, 2, 0)
        x = m.Relu(x)
        x = maxPool2d(m, x, 3, 2)
        x = Fire(m, 3, x, 96, 16, 64, 64)
        x = Fire(m, 4, x, 128, 16, 64, 64)
        x = Fire(m, 5, x, 128, 32, 128, 128)
        x = maxPool2d(m, x, 3, 2)
        x = Fire(m, 7, x, 256, 32, 128, 128)
        x = Fire(m, 8, x, 256, 48, 192, 192)
        x = Fire(m, 9, x, 384, 48, 192, 192)
        x = Fire(m, 10, x, 384, 64, 256, 256)
        x = maxPool2d(m, x, 3, 2)
        x = Fire(m, 12, x, 512, 64, 256, 256)
    case "1_1":
        x = conv2d(m, "features_0", x, 3, 64, 3, 2, 0)
        x = m.Relu(x)
        x = maxPool2d(m, x, 3, 2)
        x = Fire(m, 3, x, 64, 16, 64, 64)
        x = Fire(m, 4, x, 128, 16, 64, 64)
        x = maxPool2d(m, x, 3, 2)
        x = Fire(m, 6, x, 128, 32, 128, 128)
        x = Fire(m, 7, x, 256, 32, 128, 128)
        x = maxPool2d(m, x, 3, 2)
        x = Fire(m, 9, x, 256, 48, 192, 192)
        x = Fire(m, 10, x, 384, 48, 192, 192)
        x = Fire(m, 11, x, 384, 64, 256, 256)
        x = Fire(m, 12, x, 512, 64, 256, 256)
    }
    // classifier
    x = m.Dropout(x, "ratio", 0.5)
    // Final convolution is initialized differently from the rest
    x = conv2d(m, "classifier_1", x, 512, p.numClasses, 1, 1, 0)
    x = m.Relu(x)
    x = m.AveragePool(x, "globalPooling", true, "kernel", []int{1, 1})
    return x
}

// implementation

func Fire(
        m *models.Model, 
        id int, 
        x front.Layer,
        inPlanes int,
        squeezePlanes int,
        expand1x1Planes int,
        expand3x3Planes int) front.Layer {
    squeeze := conv2d(m, featuresId(id, "squeeze"), x, inPlanes, squeezePlanes, 1, 1, 0)
    squeeze = m.Relu(squeeze)
    expand1x1 := conv2d(m, featuresId(id, "expand1x1"), squeeze, squeezePlanes, expand1x1Planes, 1, 1, 0)
    expand1x1 = m.Relu(expand1x1)
    expand3x3 := conv2d(m, featuresId(id, "expand3x3"), squeeze, squeezePlanes, expand3x3Planes, 3, 1, 1)
    expand3x3 = m.Relu(expand3x3)
    y := m.Concat(expand1x1, expand3x3, "axis", 1)
    return y
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
    b := 
        m.Variable(
            "label", id+"_bias",
            "shape", []int{outChannels})
    return m.Conv(
        x,
        w,
        b,
        "kernel", []int{kernelSize, kernelSize},
        "stride", []int{stride, stride},
        "pads", []int{padding, padding, padding, padding})
}

// SKIPPED: ceil_mode = true (why is this needed in the original code?)

func maxPool2d(m *models.Model, x front.Layer, kernelSize int, stride int) front.Layer {
    return m.MaxPool(x, "kernel", []int{kernelSize, kernelSize}, "stride", []int{stride, stride})
}

func featuresId(id int, suffix string) string {
    return fmt.Sprintf("features_%d_%s", id, suffix) 
}

