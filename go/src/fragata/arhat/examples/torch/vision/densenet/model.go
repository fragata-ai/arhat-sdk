
package main

import "fragata/arhat/front/models"

//
//    Models
//

func DenseNet121(batchSize int, args *DenseNetArgs) *models.Model {
    // Densenet-121 model from
    // "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>
    return BuildDenseNet(batchSize, 32, []int{6, 12, 24, 16}, 64, args)
}

func DenseNet161(batchSize int, args *DenseNetArgs) *models.Model {
    // Densenet-161 model from
    // "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>
    return BuildDenseNet(batchSize, 48, []int{6, 12, 36, 24}, 96, args)
}

func DenseNet169(batchSize int, args *DenseNetArgs) *models.Model {
    // Densenet-169 model from
    // "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>
    return BuildDenseNet(batchSize, 32, []int{6, 12, 32, 32}, 64, args)
}

func DenseNet201(batchSize int, args *DenseNetArgs) *models.Model {
    // Densenet-201 model from
    // "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>
    return BuildDenseNet(batchSize, 32, []int{6, 12, 48, 32}, 64, args)
}

