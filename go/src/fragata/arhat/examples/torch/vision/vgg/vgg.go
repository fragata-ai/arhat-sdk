
package main

import (
    "fmt"
    front "fragata/arhat/front/core"
    "fragata/arhat/front/models"
)

//
//    Model builder
//

func BuildVgg(batchSize int, cfg string, batchNorm bool, numClasses int) *models.Model {
    r := NewVgg(cfg, batchNorm, numClasses)
    m := models.NewModel()
    x := m.External("shape", []int{batchSize, 3, 224, 224})
    y := r.Forward(m, x)
    y = m.Softmax(y, "axis", 1)
    m.Return(y, 0)
    return m
}

//
//    Vgg
//

type Vgg struct {
    cfg string
    batchNorm bool
    numClasses int
}

// construction/destruction

func NewVgg(cfg string, batchNorm bool, numClasses int) *Vgg {
    p := new(Vgg)
    p.cfg = cfg
    p.batchNorm = batchNorm
    p.numClasses = numClasses
    return p
}

// interface

func(p *Vgg) Forward(m *models.Model, x front.Layer) front.Layer {
    // features
    x = makeLayers(m, x, cfgs[p.cfg], p.batchNorm)
    // classifier
    x = linear(m, "classifier_0", x, []int{4096, 512, 7, 7})
    x = m.Relu(x)
    x = m.Dropout(x)
    x = linear(m, "classifier_3", x, []int{4096, 4096})
    x = m.Relu(x)
    x = m.Dropout(x)
    x = linear(m, "classifier_6", x, []int{p.numClasses, 4096})
    return x
}

// implementation

const M = -1

var cfgs = map[string][]int{
    "A": []int{64, M, 128, M, 256, 256, M, 512, 512, M, 512, 512, M},
    "B": []int{64, 64, M, 128, 128, M, 256, 256, M, 512, 512, M, 512, 512, M},
    "D": []int{64, 64, M, 128, 128, M, 256, 256, 256, M, 512, 512, 512, M, 512, 512, 512, M},
    "E": []int{64, 64, M, 128, 128, M, 256, 256, 256, 256, M, 512, 512, 512, 512, M, 512, 512, 512, 512, M},
}

func makeLayers(m *models.Model, x front.Layer, cfg []int, batchNorm bool) front.Layer {
    inChannels := 3
    id := 0
    for _, v := range cfg {
        if v == M {
            x = m.MaxPool(x, "kernel", []int{2, 2}, "stride", []int{2, 2})
            id++
        } else {
            x = conv2d(m, featuresId(id), x, inChannels, v, 3, 1)
            id++
            if batchNorm {
                x = batchNorm2d(m, featuresId(id), x, v)
                id++
            }
            x = m.Relu(x)
            id++
            inChannels = v
        }
    }
    return x
}

func conv2d(
        m *models.Model, 
        id string, 
        x front.Layer, 
        inChannels int, 
        outChannels int, 
        kernelSize int, 
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

func featuresId(id int) string {
    return fmt.Sprintf("features_%d", id) 
}

