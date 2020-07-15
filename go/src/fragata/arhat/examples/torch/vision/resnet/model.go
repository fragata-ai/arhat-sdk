
package main

import "fragata/arhat/front/models"

//
//    Models
//

func ResNet18(batchSize int, args *ResNetArgs) *models.Model {
    // ResNet-18 model from
    // "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    return BuildResNet(batchSize, NewBasicBlock(), []int{2, 2, 2, 2}, args)
}

func ResNet34(batchSize int, args *ResNetArgs) *models.Model {
    // ResNet-34 model from
    // "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    return BuildResNet(batchSize, NewBasicBlock(), []int{3, 4, 6, 3}, args)
}

func ResNet50(batchSize int, args *ResNetArgs) *models.Model {
    // ResNet-50 model from
    // "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    return BuildResNet(batchSize, NewBottleneck(), []int{3, 4, 6, 3}, args)
}

func ResNet101(batchSize int, args *ResNetArgs) *models.Model {
    // ResNet-101 model from
    // "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    return BuildResNet(batchSize, NewBottleneck(), []int{3, 4, 23, 3}, args)
}

func ResNet152(batchSize int, args *ResNetArgs) *models.Model {
    // ResNet-152 model from
    // "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    return BuildResNet(batchSize, NewBottleneck(), []int{3, 8, 36, 3}, args)
}

func ResNeXt50_32x4d(batchSize int, args *ResNetArgs) *models.Model {
    // ResNeXt-50 32x4d model from
    // "Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>
    if args == nil {
        args = NewResNetArgs()
    }
    args.SetGroups(32)
    args.SetWidthPerGroup(4)
    return BuildResNet(batchSize, NewBottleneck(), []int{3, 4, 6, 3}, args)
}

func ResNeXt101_32x8d(batchSize int, args *ResNetArgs) *models.Model {
    // ResNeXt-101 32x8d model from
    // "Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>
    if args == nil {
        args = NewResNetArgs()
    }
    args.SetGroups(32)
    args.SetWidthPerGroup(8)
    return BuildResNet(batchSize, NewBottleneck(), []int{3, 4, 23, 3}, args)
}

func WideResNet50_2(batchSize int, args *ResNetArgs) *models.Model {
    // Wide ResNet-50-2 model from
    // "Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>
    //
    // The model is the same as ResNet except for the bottleneck number of channels
    // which is twice larger in every block. The number of channels in outer 1x1
    // convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    // channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    if args == nil {
        args = NewResNetArgs()
    }
    args.SetWidthPerGroup(64*2)
    return BuildResNet(batchSize, NewBottleneck(), []int{3, 4, 6, 3}, args)
}

func WideResNet101_2(batchSize int, args *ResNetArgs) *models.Model {
    // Wide ResNet-101-2 model from
    // "Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>
    //
    // The model is the same as ResNet except for the bottleneck number of channels
    // which is twice larger in every block. The number of channels in outer 1x1
    // convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    // channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    if args == nil {
        args = NewResNetArgs()
    }
    args.SetWidthPerGroup(64*2)
    return BuildResNet(batchSize, NewBottleneck(), []int{3, 4, 23, 3}, args)
}

