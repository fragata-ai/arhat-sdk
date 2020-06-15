
package main

import "fragata/arhat/front/models"

//
//    AlexnetCaffe
//

func AlexnetCaffe() *models.Model {
    m := models.NewModel()
    variable := m.Variable("label", "conv1_blob1", "shape", []int{96, 3, 11, 11})
    variable1 := m.Variable("label", "conv1_blob2", "shape", []int{96})
    variable2 := m.Variable("label", "conv2_blob1", "shape", []int{256, 48, 5, 5})
    variable3 := m.Variable("label", "conv2_blob2", "shape", []int{256})
    variable4 := m.Variable("label", "conv3_blob1", "shape", []int{384, 256, 3, 3})
    variable5 := m.Variable("label", "conv3_blob2", "shape", []int{384})
    variable6 := m.Variable("label", "conv4_blob1", "shape", []int{384, 192, 3, 3})
    variable7 := m.Variable("label", "conv4_blob2", "shape", []int{384})
    variable8 := m.Variable("label", "conv5_blob1", "shape", []int{256, 192, 3, 3})
    variable9 := m.Variable("label", "conv5_blob2", "shape", []int{256})
    variable10 := m.Variable("label", "fc6_blob1", "shape", []int{4096, 9216})
    variable11 := m.Variable("label", "fc6_blob2", "shape", []int{4096})
    variable12 := m.Variable("label", "fc7_blob1", "shape", []int{4096, 4096})
    variable13 := m.Variable("label", "fc7_blob2", "shape", []int{4096})
    variable14 := m.Variable("label", "fc8_blob1", "shape", []int{1000, 4096})
    variable15 := m.Variable("label", "fc8_blob2", "shape", []int{1000})
    data := m.External("shape", []int{10, 3, 227, 227})
    conv := m.Conv(data, variable, variable1, "kernel", []int{11, 11}, "stride", []int{4, 4})
    relu := m.Relu(conv)
    lrn := m.Lrn(relu, "size", 5, "alpha", 1.0e-04, "beta", 0.75, "bias", 1.0)
    maxPool := m.MaxPool(lrn, "kernel", []int{3, 3}, "stride", []int{2, 2})
    conv1 := 
        m.Conv(maxPool, variable2, variable3, 
            "kernel", []int{5, 5}, "pads", []int{2, 2, 2, 2}, "group", 2)
    relu1 := m.Relu(conv1)
    lrn1 := m.Lrn(relu1, "size", 5, "alpha", 1.0e-04, "beta", 0.75, "bias", 1.0)
    maxPool1 := m.MaxPool(lrn1, "kernel", []int{3, 3}, "stride", []int{2, 2})
    conv2 := 
        m.Conv(maxPool1, variable4, variable5, 
            "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    relu2 := m.Relu(conv2)
    conv3 := 
        m.Conv(relu2, variable6, variable7, 
            "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1}, "group", 2)
    relu3 := m.Relu(conv3)
    conv4 := 
        m.Conv(relu3, variable8, variable9, 
            "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1}, "group", 2)
    relu4 := m.Relu(conv4)
    maxPool2 := m.MaxPool(relu4, "kernel", []int{3, 3}, "stride", []int{2, 2})
    reshape := m.Reshape(maxPool2, "shape", []int{10, -1})
    linear := m.FullyConnected(reshape, variable10, variable11)
    relu5 := m.Relu(linear)
    linear1 := m.FullyConnected(relu5, variable12, variable13)
    relu6 := m.Relu(linear1)
    linear2 := m.FullyConnected(relu6, variable14, variable15)
    prob := m.Softmax(linear2, "axis", 1)
    m.Return(prob, 0)
    return m
}

