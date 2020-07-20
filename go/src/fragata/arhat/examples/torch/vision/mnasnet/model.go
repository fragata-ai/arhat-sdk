
package main

import "fragata/arhat/front/models"

//
//    Models
//

func MnasNet05(batchSize int, args *MnasNetArgs) *models.Model {
    // MNASNet with depth multiplier of 0.5 from
    // "MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    // <https://arxiv.org/pdf/1807.11626.pdf>
    return BuildMnasNet(batchSize, 0.5, args)
}

func MnasNet075(batchSize int, args *MnasNetArgs) *models.Model {
    // MNASNet with depth multiplier of 0.75 from
    // "MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    // <https://arxiv.org/pdf/1807.11626.pdf>
    return BuildMnasNet(batchSize, 0.75, args)
}

func MnasNet10(batchSize int, args *MnasNetArgs) *models.Model {
    // MNASNet with depth multiplier of 1.0 from
    // "MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    // <https://arxiv.org/pdf/1807.11626.pdf>
    return BuildMnasNet(batchSize, 1.0, args)
}

func MnasNet13(batchSize int, args *MnasNetArgs) *models.Model {
    // MNASNet with depth multiplier of 1.3 from
    // "MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    // <https://arxiv.org/pdf/1807.11626.pdf>
    return BuildMnasNet(batchSize, 1.3, args)
}

