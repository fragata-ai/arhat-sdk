
package main

import "fragata/arhat/front/models"

//
//    Models
//

func ShuffleNetV2x05(batchSize int, args *ShuffleNetV2Args) *models.Model {
    // Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    // "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    // <https://arxiv.org/abs/1807.11164>
    return BuildShuffleNetV2(batchSize,  []int{4, 8, 4}, []int{24, 48, 96, 192, 1024}, args)
}

func ShuffleNetV2x10(batchSize int, args *ShuffleNetV2Args) *models.Model {
    // Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    // "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    // <https://arxiv.org/abs/1807.11164>
    return BuildShuffleNetV2(batchSize,  []int{4, 8, 4}, []int{24, 116, 232, 464, 1024}, args)
}

func ShuffleNetV2x15(batchSize int, args *ShuffleNetV2Args) *models.Model {
    // Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    // "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    // <https://arxiv.org/abs/1807.11164>
    return BuildShuffleNetV2(batchSize,  []int{4, 8, 4}, []int{24, 176, 352, 704, 1024}, args)
}

func ShuffleNetV2x20(batchSize int, args *ShuffleNetV2Args) *models.Model {
    // Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    // "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    // <https://arxiv.org/abs/1807.11164>
    return BuildShuffleNetV2(batchSize,  []int{4, 8, 4}, []int{24, 244, 488, 976, 2048}, args)
}

