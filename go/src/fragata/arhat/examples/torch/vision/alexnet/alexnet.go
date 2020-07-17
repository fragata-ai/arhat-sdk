
package main

import (
    front "fragata/arhat/front/core"
    "fragata/arhat/front/models"
)

//
//    Model builder
//

// AlexNet model architecture from the
// "One weird trick..." <https://arxiv.org/abs/1404.5997> paper. 

func BuildAlexNet(batchSize int, numClasses int) *models.Model {
    r := NewAlexNet(numClasses)
    m := models.NewModel()
    x := m.External("shape", []int{batchSize, 3, 224, 224})
    y := r.Forward(m, x)
    y = m.Softmax(y, "axis", 1)
    m.Return(y, 0)
    return m
}

//
//    AlexNet
//

type AlexNet struct {
    numClasses int
}

// construction/destruction

func NewAlexNet(numClasses int) *AlexNet {
    p := new(AlexNet)
    p.numClasses = numClasses
    return p
}

// interface

func(p *AlexNet) Forward(m *models.Model, x front.Layer) front.Layer {
    // features
    x = conv2d(m, "features_0", x, 3, 64, 11, 4, 2)
    x = m.Relu(x)
    x = maxPool2d(m, x, 3, 2)
    x = conv2d(m, "features_3", x, 64, 192, 5, 1, 2)
    x = m.Relu(x)
    x = maxPool2d(m, x, 3, 2)
    x = conv2d(m, "features_6", x, 192, 384, 3, 1, 1)
    x = m.Relu(x)
    x = conv2d(m, "features_8", x, 384, 256, 3, 1, 1)
    x = m.Relu(x)
    x = conv2d(m, "features_10", x, 256, 256, 3, 1, 1)
    x = m.Relu(x)
    x = maxPool2d(m, x, 3, 2)
    // classifier
    x = m.Dropout(x)
    x = linear(m, "classifier_1", x, []int{4096, 256, 6, 6})
    x = m.Relu(x)
    x = m.Dropout(x)
    x = linear(m, "classifier_4", x, []int{4096, 4096})
    x = m.Relu(x)
    x = linear(m, "classifier_6", x, []int{p.numClasses, 4096}) 
    return x
}

// implementation

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

func maxPool2d(m *models.Model, x front.Layer, kernelSize int, stride int) front.Layer {
    return m.MaxPool(x, "kernel", []int{kernelSize, kernelSize}, "stride", []int{stride, stride})
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

