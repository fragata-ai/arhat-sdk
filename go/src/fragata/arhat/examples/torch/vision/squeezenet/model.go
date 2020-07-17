
package main

import "fragata/arhat/front/models"

//
//    Models
//

func SqueezeNet10(batchSize int, numClasses int) *models.Model {
    // SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    // accuracy with 50x fewer parameters and <0.5MB model size"
    // <https://arxiv.org/abs/1602.07360> paper.
    return BuildSqueezeNet(batchSize, "1_0", numClasses)
}

func SqueezeNet11(batchSize int, numClasses int) *models.Model {
    // SqueezeNet 1.1 model from the `official SqueezeNet repo
    // <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>
    // SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    // than SqueezeNet 1.0, without sacrificing accuracy.
    return BuildSqueezeNet(batchSize, "1_1", numClasses)
}

