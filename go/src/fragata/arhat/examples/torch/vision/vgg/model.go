
package main

import "fragata/arhat/front/models"

//
//    Models
//

func Vgg11(batchSize int, numClasses int) *models.Model {
    // VGG 11-layer model (configuration "A") from
    // "Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>
    return BuildVgg(batchSize, "A", false, numClasses)
}

func Vgg11Bn(batchSize int, numClasses int) *models.Model {
    // VGG 11-layer model (configuration "A") with batch normalization from
    // "Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>
    return BuildVgg(batchSize, "A", true, numClasses)
}

func Vgg13(batchSize int, numClasses int) *models.Model {
    // VGG 13-layer model (configuration "B") from
    // "Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>
    return BuildVgg(batchSize, "B", false, numClasses)
}

func Vgg13Bn(batchSize int, numClasses int) *models.Model {
    // VGG 13-layer model (configuration "B") from
    // "Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>
    return BuildVgg(batchSize, "B", true, numClasses)
}

func Vgg16(batchSize int, numClasses int) *models.Model {
    // VGG 16-layer model (configuration "D") from
    // "Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>
    return BuildVgg(batchSize, "D", false, numClasses)
}

func Vgg16Bn(batchSize int, numClasses int) *models.Model {
    // VGG 16-layer model (configuration "D") with batch normalization from
    // "Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>
    return BuildVgg(batchSize, "D", true, numClasses)
}

func Vgg19(batchSize int, numClasses int) *models.Model {
    // VGG 19-layer model (configuration "E") from
    // "Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>
    return BuildVgg(batchSize, "E", false, numClasses)
}

func Vgg19Bn(batchSize int, numClasses int) *models.Model {
    // VGG 19-layer model (configuration "E") with batch normalization from
    // "Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>
    return BuildVgg(batchSize, "E", true, numClasses)
}

