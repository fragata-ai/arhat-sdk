
package main

import "fragata/arhat/front/models"

//
//    GoogleNetCaffe
//

func GoogleNetCaffe() *models.Model {
    m := models.NewModel()
    external1 := m.External("shape", []int{10, 3, 224, 224})
    variable1 := m.Variable("label", "conv1_7x7_s2_blob1", "shape", []int{64, 3, 7, 7})
    variable2 := m.Variable("label", "conv1_7x7_s2_blob2", "shape", []int{64})
    variable3 := m.Variable("label", "conv2_3x3_reduce_blob1", "shape", []int{64, 64, 1, 1})
    variable4 := m.Variable("label", "conv2_3x3_reduce_blob2", "shape", []int{64})
    variable5 := m.Variable("label", "conv2_3x3_blob1", "shape", []int{192, 64, 3, 3})
    variable6 := m.Variable("label", "conv2_3x3_blob2", "shape", []int{192})
    variable7 := m.Variable("label", "inception_3a_1x1_blob1", "shape", []int{64, 192, 1, 1})
    variable8 := m.Variable("label", "inception_3a_1x1_blob2", "shape", []int{64})
    variable9 := m.Variable("label", "inception_3a_3x3_reduce_blob1", "shape", []int{96, 192, 1, 1})
    variable10 := m.Variable("label", "inception_3a_3x3_reduce_blob2", "shape", []int{96})
    variable11 := m.Variable("label", "inception_3a_3x3_blob1", "shape", []int{128, 96, 3, 3})
    variable12 := m.Variable("label", "inception_3a_3x3_blob2", "shape", []int{128})
    variable13 := m.Variable("label", "inception_3a_5x5_reduce_blob1", "shape", []int{16, 192, 1, 1})
    variable14 := m.Variable("label", "inception_3a_5x5_reduce_blob2", "shape", []int{16})
    variable15 := m.Variable("label", "inception_3a_5x5_blob1", "shape", []int{32, 16, 5, 5})
    variable16 := m.Variable("label", "inception_3a_5x5_blob2", "shape", []int{32})
    variable17 := m.Variable("label", "inception_3a_pool_proj_blob1", "shape", []int{32, 192, 1, 1})
    variable18 := m.Variable("label", "inception_3a_pool_proj_blob2", "shape", []int{32})
    variable19 := m.Variable("label", "inception_3b_1x1_blob1", "shape", []int{128, 256, 1, 1})
    variable20 := m.Variable("label", "inception_3b_1x1_blob2", "shape", []int{128})
    variable21 := m.Variable("label", "inception_3b_3x3_reduce_blob1", "shape", []int{128, 256, 1, 1})
    variable22 := m.Variable("label", "inception_3b_3x3_reduce_blob2", "shape", []int{128})
    variable23 := m.Variable("label", "inception_3b_3x3_blob1", "shape", []int{192, 128, 3, 3})
    variable24 := m.Variable("label", "inception_3b_3x3_blob2", "shape", []int{192})
    variable25 := m.Variable("label", "inception_3b_5x5_reduce_blob1", "shape", []int{32, 256, 1, 1})
    variable26 := m.Variable("label", "inception_3b_5x5_reduce_blob2", "shape", []int{32})
    variable27 := m.Variable("label", "inception_3b_5x5_blob1", "shape", []int{96, 32, 5, 5})
    variable28 := m.Variable("label", "inception_3b_5x5_blob2", "shape", []int{96})
    variable29 := m.Variable("label", "inception_3b_pool_proj_blob1", "shape", []int{64, 256, 1, 1})
    variable30 := m.Variable("label", "inception_3b_pool_proj_blob2", "shape", []int{64})
    variable31 := m.Variable("label", "inception_4a_1x1_blob1", "shape", []int{192, 480, 1, 1})
    variable32 := m.Variable("label", "inception_4a_1x1_blob2", "shape", []int{192})
    variable33 := m.Variable("label", "inception_4a_3x3_reduce_blob1", "shape", []int{96, 480, 1, 1})
    variable34 := m.Variable("label", "inception_4a_3x3_reduce_blob2", "shape", []int{96})
    variable35 := m.Variable("label", "inception_4a_3x3_blob1", "shape", []int{208, 96, 3, 3})
    variable36 := m.Variable("label", "inception_4a_3x3_blob2", "shape", []int{208})
    variable37 := m.Variable("label", "inception_4a_5x5_reduce_blob1", "shape", []int{16, 480, 1, 1})
    variable38 := m.Variable("label", "inception_4a_5x5_reduce_blob2", "shape", []int{16})
    variable39 := m.Variable("label", "inception_4a_5x5_blob1", "shape", []int{48, 16, 5, 5})
    variable40 := m.Variable("label", "inception_4a_5x5_blob2", "shape", []int{48})
    variable41 := m.Variable("label", "inception_4a_pool_proj_blob1", "shape", []int{64, 480, 1, 1})
    variable42 := m.Variable("label", "inception_4a_pool_proj_blob2", "shape", []int{64})
    variable43 := m.Variable("label", "inception_4b_1x1_blob1", "shape", []int{160, 512, 1, 1})
    variable44 := m.Variable("label", "inception_4b_1x1_blob2", "shape", []int{160})
    variable45 := m.Variable("label", "inception_4b_3x3_reduce_blob1", "shape", []int{112, 512, 1, 1})
    variable46 := m.Variable("label", "inception_4b_3x3_reduce_blob2", "shape", []int{112})
    variable47 := m.Variable("label", "inception_4b_3x3_blob1", "shape", []int{224, 112, 3, 3})
    variable48 := m.Variable("label", "inception_4b_3x3_blob2", "shape", []int{224})
    variable49 := m.Variable("label", "inception_4b_5x5_reduce_blob1", "shape", []int{24, 512, 1, 1})
    variable50 := m.Variable("label", "inception_4b_5x5_reduce_blob2", "shape", []int{24})
    variable51 := m.Variable("label", "inception_4b_5x5_blob1", "shape", []int{64, 24, 5, 5})
    variable52 := m.Variable("label", "inception_4b_5x5_blob2", "shape", []int{64})
    variable53 := m.Variable("label", "inception_4b_pool_proj_blob1", "shape", []int{64, 512, 1, 1})
    variable54 := m.Variable("label", "inception_4b_pool_proj_blob2", "shape", []int{64})
    variable55 := m.Variable("label", "inception_4c_1x1_blob1", "shape", []int{128, 512, 1, 1})
    variable56 := m.Variable("label", "inception_4c_1x1_blob2", "shape", []int{128})
    variable57 := m.Variable("label", "inception_4c_3x3_reduce_blob1", "shape", []int{128, 512, 1, 1})
    variable58 := m.Variable("label", "inception_4c_3x3_reduce_blob2", "shape", []int{128})
    variable59 := m.Variable("label", "inception_4c_3x3_blob1", "shape", []int{256, 128, 3, 3})
    variable60 := m.Variable("label", "inception_4c_3x3_blob2", "shape", []int{256})
    variable61 := m.Variable("label", "inception_4c_5x5_reduce_blob1", "shape", []int{24, 512, 1, 1})
    variable62 := m.Variable("label", "inception_4c_5x5_reduce_blob2", "shape", []int{24})
    variable63 := m.Variable("label", "inception_4c_5x5_blob1", "shape", []int{64, 24, 5, 5})
    variable64 := m.Variable("label", "inception_4c_5x5_blob2", "shape", []int{64})
    variable65 := m.Variable("label", "inception_4c_pool_proj_blob1", "shape", []int{64, 512, 1, 1})
    variable66 := m.Variable("label", "inception_4c_pool_proj_blob2", "shape", []int{64})
    variable67 := m.Variable("label", "inception_4d_1x1_blob1", "shape", []int{112, 512, 1, 1})
    variable68 := m.Variable("label", "inception_4d_1x1_blob2", "shape", []int{112})
    variable69 := m.Variable("label", "inception_4d_3x3_reduce_blob1", "shape", []int{144, 512, 1, 1})
    variable70 := m.Variable("label", "inception_4d_3x3_reduce_blob2", "shape", []int{144})
    variable71 := m.Variable("label", "inception_4d_3x3_blob1", "shape", []int{288, 144, 3, 3})
    variable72 := m.Variable("label", "inception_4d_3x3_blob2", "shape", []int{288})
    variable73 := m.Variable("label", "inception_4d_5x5_reduce_blob1", "shape", []int{32, 512, 1, 1})
    variable74 := m.Variable("label", "inception_4d_5x5_reduce_blob2", "shape", []int{32})
    variable75 := m.Variable("label", "inception_4d_5x5_blob1", "shape", []int{64, 32, 5, 5})
    variable76 := m.Variable("label", "inception_4d_5x5_blob2", "shape", []int{64})
    variable77 := m.Variable("label", "inception_4d_pool_proj_blob1", "shape", []int{64, 512, 1, 1})
    variable78 := m.Variable("label", "inception_4d_pool_proj_blob2", "shape", []int{64})
    variable79 := m.Variable("label", "inception_4e_1x1_blob1", "shape", []int{256, 528, 1, 1})
    variable80 := m.Variable("label", "inception_4e_1x1_blob2", "shape", []int{256})
    variable81 := m.Variable("label", "inception_4e_3x3_reduce_blob1", "shape", []int{160, 528, 1, 1})
    variable82 := m.Variable("label", "inception_4e_3x3_reduce_blob2", "shape", []int{160})
    variable83 := m.Variable("label", "inception_4e_3x3_blob1", "shape", []int{320, 160, 3, 3})
    variable84 := m.Variable("label", "inception_4e_3x3_blob2", "shape", []int{320})
    variable85 := m.Variable("label", "inception_4e_5x5_reduce_blob1", "shape", []int{32, 528, 1, 1})
    variable86 := m.Variable("label", "inception_4e_5x5_reduce_blob2", "shape", []int{32})
    variable87 := m.Variable("label", "inception_4e_5x5_blob1", "shape", []int{128, 32, 5, 5})
    variable88 := m.Variable("label", "inception_4e_5x5_blob2", "shape", []int{128})
    variable89 := m.Variable("label", "inception_4e_pool_proj_blob1", "shape", []int{128, 528, 1, 1})
    variable90 := m.Variable("label", "inception_4e_pool_proj_blob2", "shape", []int{128})
    variable91 := m.Variable("label", "inception_5a_1x1_blob1", "shape", []int{256, 832, 1, 1})
    variable92 := m.Variable("label", "inception_5a_1x1_blob2", "shape", []int{256})
    variable93 := m.Variable("label", "inception_5a_3x3_reduce_blob1", "shape", []int{160, 832, 1, 1})
    variable94 := m.Variable("label", "inception_5a_3x3_reduce_blob2", "shape", []int{160})
    variable95 := m.Variable("label", "inception_5a_3x3_blob1", "shape", []int{320, 160, 3, 3})
    variable96 := m.Variable("label", "inception_5a_3x3_blob2", "shape", []int{320})
    variable97 := m.Variable("label", "inception_5a_5x5_reduce_blob1", "shape", []int{32, 832, 1, 1})
    variable98 := m.Variable("label", "inception_5a_5x5_reduce_blob2", "shape", []int{32})
    variable99 := m.Variable("label", "inception_5a_5x5_blob1", "shape", []int{128, 32, 5, 5})
    variable100 := m.Variable("label", "inception_5a_5x5_blob2", "shape", []int{128})
    variable101 := m.Variable("label", "inception_5a_pool_proj_blob1", "shape", []int{128, 832, 1, 1})
    variable102 := m.Variable("label", "inception_5a_pool_proj_blob2", "shape", []int{128})
    variable103 := m.Variable("label", "inception_5b_1x1_blob1", "shape", []int{384, 832, 1, 1})
    variable104 := m.Variable("label", "inception_5b_1x1_blob2", "shape", []int{384})
    variable105 := m.Variable("label", "inception_5b_3x3_reduce_blob1", "shape", []int{192, 832, 1, 1})
    variable106 := m.Variable("label", "inception_5b_3x3_reduce_blob2", "shape", []int{192})
    variable107 := m.Variable("label", "inception_5b_3x3_blob1", "shape", []int{384, 192, 3, 3})
    variable108 := m.Variable("label", "inception_5b_3x3_blob2", "shape", []int{384})
    variable109 := m.Variable("label", "inception_5b_5x5_reduce_blob1", "shape", []int{48, 832, 1, 1})
    variable110 := m.Variable("label", "inception_5b_5x5_reduce_blob2", "shape", []int{48})
    variable111 := m.Variable("label", "inception_5b_5x5_blob1", "shape", []int{128, 48, 5, 5})
    variable112 := m.Variable("label", "inception_5b_5x5_blob2", "shape", []int{128})
    variable113 := m.Variable("label", "inception_5b_pool_proj_blob1", "shape", []int{128, 832, 1, 1})
    variable114 := m.Variable("label", "inception_5b_pool_proj_blob2", "shape", []int{128})
    variable115 := m.Variable("label", "loss3_classifier_blob1", "shape", []int{1000, 1024})
    variable116 := m.Variable("label", "loss3_classifier_blob2", "shape", []int{1000})
    conv1 := m.Conv(external1, variable1, variable2, "kernel", []int{7, 7}, "stride", []int{2, 2}, "pads", []int{3, 3, 3, 3})
    relu1 := m.Relu(conv1)
    maxPool1 := m.MaxPool(relu1, "kernel", []int{3, 3}, "stride", []int{2, 2}, "pads", []int{0, 0, 1, 1})
    lrn1 := m.Lrn(maxPool1, "size", 5, "alpha", 0.0001, "beta", 0.75)
    conv2 := m.Conv(lrn1, variable3, variable4, "kernel", []int{1, 1})
    relu2 := m.Relu(conv2)
    conv3 := m.Conv(relu2, variable5, variable6, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    relu3 := m.Relu(conv3)
    lrn2 := m.Lrn(relu3, "size", 5, "alpha", 0.0001, "beta", 0.75)
    maxPool2 := m.MaxPool(lrn2, "kernel", []int{3, 3}, "stride", []int{2, 2}, "pads", []int{0, 0, 1, 1})
    maxPool3 := m.MaxPool(maxPool2, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    conv4 := m.Conv(maxPool3, variable17, variable18, "kernel", []int{1, 1})
    relu4 := m.Relu(conv4)
    conv5 := m.Conv(maxPool2, variable13, variable14, "kernel", []int{1, 1})
    relu5 := m.Relu(conv5)
    conv6 := m.Conv(relu5, variable15, variable16, "kernel", []int{5, 5}, "pads", []int{2, 2, 2, 2})
    relu6 := m.Relu(conv6)
    conv7 := m.Conv(maxPool2, variable9, variable10, "kernel", []int{1, 1})
    relu7 := m.Relu(conv7)
    conv8 := m.Conv(relu7, variable11, variable12, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    relu8 := m.Relu(conv8)
    conv9 := m.Conv(maxPool2, variable7, variable8, "kernel", []int{1, 1})
    relu9 := m.Relu(conv9)
    concat1 := m.Concat(relu9, relu8, relu6, relu4, "axis", 1)
    maxPool4 := m.MaxPool(concat1, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    conv10 := m.Conv(maxPool4, variable29, variable30, "kernel", []int{1, 1})
    relu10 := m.Relu(conv10)
    conv11 := m.Conv(concat1, variable25, variable26, "kernel", []int{1, 1})
    relu11 := m.Relu(conv11)
    conv12 := m.Conv(relu11, variable27, variable28, "kernel", []int{5, 5}, "pads", []int{2, 2, 2, 2})
    relu12 := m.Relu(conv12)
    conv13 := m.Conv(concat1, variable21, variable22, "kernel", []int{1, 1})
    relu13 := m.Relu(conv13)
    conv14 := m.Conv(relu13, variable23, variable24, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    relu14 := m.Relu(conv14)
    conv15 := m.Conv(concat1, variable19, variable20, "kernel", []int{1, 1})
    relu15 := m.Relu(conv15)
    concat2 := m.Concat(relu15, relu14, relu12, relu10, "axis", 1)
    maxPool5 := m.MaxPool(concat2, "kernel", []int{3, 3}, "stride", []int{2, 2}, "pads", []int{0, 0, 1, 1})
    maxPool6 := m.MaxPool(maxPool5, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    conv16 := m.Conv(maxPool6, variable41, variable42, "kernel", []int{1, 1})
    relu16 := m.Relu(conv16)
    conv17 := m.Conv(maxPool5, variable37, variable38, "kernel", []int{1, 1})
    relu17 := m.Relu(conv17)
    conv18 := m.Conv(relu17, variable39, variable40, "kernel", []int{5, 5}, "pads", []int{2, 2, 2, 2})
    relu18 := m.Relu(conv18)
    conv19 := m.Conv(maxPool5, variable33, variable34, "kernel", []int{1, 1})
    relu19 := m.Relu(conv19)
    conv20 := m.Conv(relu19, variable35, variable36, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    relu20 := m.Relu(conv20)
    conv21 := m.Conv(maxPool5, variable31, variable32, "kernel", []int{1, 1})
    relu21 := m.Relu(conv21)
    concat3 := m.Concat(relu21, relu20, relu18, relu16, "axis", 1)
    maxPool7 := m.MaxPool(concat3, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    conv22 := m.Conv(maxPool7, variable53, variable54, "kernel", []int{1, 1})
    relu22 := m.Relu(conv22)
    conv23 := m.Conv(concat3, variable49, variable50, "kernel", []int{1, 1})
    relu23 := m.Relu(conv23)
    conv24 := m.Conv(relu23, variable51, variable52, "kernel", []int{5, 5}, "pads", []int{2, 2, 2, 2})
    relu24 := m.Relu(conv24)
    conv25 := m.Conv(concat3, variable45, variable46, "kernel", []int{1, 1})
    relu25 := m.Relu(conv25)
    conv26 := m.Conv(relu25, variable47, variable48, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    relu26 := m.Relu(conv26)
    conv27 := m.Conv(concat3, variable43, variable44, "kernel", []int{1, 1})
    relu27 := m.Relu(conv27)
    concat4 := m.Concat(relu27, relu26, relu24, relu22, "axis", 1)
    maxPool8 := m.MaxPool(concat4, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    conv28 := m.Conv(maxPool8, variable65, variable66, "kernel", []int{1, 1})
    relu28 := m.Relu(conv28)
    conv29 := m.Conv(concat4, variable61, variable62, "kernel", []int{1, 1})
    relu29 := m.Relu(conv29)
    conv30 := m.Conv(relu29, variable63, variable64, "kernel", []int{5, 5}, "pads", []int{2, 2, 2, 2})
    relu30 := m.Relu(conv30)
    conv31 := m.Conv(concat4, variable57, variable58, "kernel", []int{1, 1})
    relu31 := m.Relu(conv31)
    conv32 := m.Conv(relu31, variable59, variable60, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    relu32 := m.Relu(conv32)
    conv33 := m.Conv(concat4, variable55, variable56, "kernel", []int{1, 1})
    relu33 := m.Relu(conv33)
    concat5 := m.Concat(relu33, relu32, relu30, relu28, "axis", 1)
    maxPool9 := m.MaxPool(concat5, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    conv34 := m.Conv(maxPool9, variable77, variable78, "kernel", []int{1, 1})
    relu34 := m.Relu(conv34)
    conv35 := m.Conv(concat5, variable73, variable74, "kernel", []int{1, 1})
    relu35 := m.Relu(conv35)
    conv36 := m.Conv(relu35, variable75, variable76, "kernel", []int{5, 5}, "pads", []int{2, 2, 2, 2})
    relu36 := m.Relu(conv36)
    conv37 := m.Conv(concat5, variable69, variable70, "kernel", []int{1, 1})
    relu37 := m.Relu(conv37)
    conv38 := m.Conv(relu37, variable71, variable72, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    relu38 := m.Relu(conv38)
    conv39 := m.Conv(concat5, variable67, variable68, "kernel", []int{1, 1})
    relu39 := m.Relu(conv39)
    concat6 := m.Concat(relu39, relu38, relu36, relu34, "axis", 1)
    maxPool10 := m.MaxPool(concat6, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    conv40 := m.Conv(maxPool10, variable89, variable90, "kernel", []int{1, 1})
    relu40 := m.Relu(conv40)
    conv41 := m.Conv(concat6, variable85, variable86, "kernel", []int{1, 1})
    relu41 := m.Relu(conv41)
    conv42 := m.Conv(relu41, variable87, variable88, "kernel", []int{5, 5}, "pads", []int{2, 2, 2, 2})
    relu42 := m.Relu(conv42)
    conv43 := m.Conv(concat6, variable81, variable82, "kernel", []int{1, 1})
    relu43 := m.Relu(conv43)
    conv44 := m.Conv(relu43, variable83, variable84, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    relu44 := m.Relu(conv44)
    conv45 := m.Conv(concat6, variable79, variable80, "kernel", []int{1, 1})
    relu45 := m.Relu(conv45)
    concat7 := m.Concat(relu45, relu44, relu42, relu40, "axis", 1)
    maxPool11 := m.MaxPool(concat7, "kernel", []int{3, 3}, "stride", []int{2, 2}, "pads", []int{0, 0, 1, 1})
    maxPool12 := m.MaxPool(maxPool11, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    conv46 := m.Conv(maxPool12, variable101, variable102, "kernel", []int{1, 1})
    relu46 := m.Relu(conv46)
    conv47 := m.Conv(maxPool11, variable97, variable98, "kernel", []int{1, 1})
    relu47 := m.Relu(conv47)
    conv48 := m.Conv(relu47, variable99, variable100, "kernel", []int{5, 5}, "pads", []int{2, 2, 2, 2})
    relu48 := m.Relu(conv48)
    conv49 := m.Conv(maxPool11, variable93, variable94, "kernel", []int{1, 1})
    relu49 := m.Relu(conv49)
    conv50 := m.Conv(relu49, variable95, variable96, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    relu50 := m.Relu(conv50)
    conv51 := m.Conv(maxPool11, variable91, variable92, "kernel", []int{1, 1})
    relu51 := m.Relu(conv51)
    concat8 := m.Concat(relu51, relu50, relu48, relu46, "axis", 1)
    maxPool13 := m.MaxPool(concat8, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    conv52 := m.Conv(maxPool13, variable113, variable114, "kernel", []int{1, 1})
    relu52 := m.Relu(conv52)
    conv53 := m.Conv(concat8, variable109, variable110, "kernel", []int{1, 1})
    relu53 := m.Relu(conv53)
    conv54 := m.Conv(relu53, variable111, variable112, "kernel", []int{5, 5}, "pads", []int{2, 2, 2, 2})
    relu54 := m.Relu(conv54)
    conv55 := m.Conv(concat8, variable105, variable106, "kernel", []int{1, 1})
    relu55 := m.Relu(conv55)
    conv56 := m.Conv(relu55, variable107, variable108, "kernel", []int{3, 3}, "pads", []int{1, 1, 1, 1})
    relu56 := m.Relu(conv56)
    conv57 := m.Conv(concat8, variable103, variable104, "kernel", []int{1, 1})
    relu57 := m.Relu(conv57)
    concat9 := m.Concat(relu57, relu56, relu54, relu52, "axis", 1)
    avgPool1 := m.AveragePool(concat9, "kernel", []int{7, 7}, "countIncludePad", true)
    reshape1 := m.Reshape(avgPool1, "shape", []int{10, -1})
    linear1 := m.FullyConnected(reshape1, variable115, variable116)
    softmax1 := m.Softmax(linear1, "axis", 1)
    m.Return(softmax1, 0)
    return m
}

