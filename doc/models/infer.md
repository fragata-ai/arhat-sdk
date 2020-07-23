
# Getting started with Arhat models

Arhat SDK is shipped with a collection of pre-trained deep learning models.
These models are available in form of Go source code and as pre-fabricated
C++ source code produced by Arhat for the supported target architectures.
This tutorial contains step by step introduction into using this models
for inference. We will start with building models from the pre-fabricated
C++ source code and using then for inference with a pre-fabricated data set,
then proceed with instructions on ingesting user-supplied data sets, and
finally will demonstrate how to use Arhat for generation of platform-specific
C++ code from the original Go descriptions.

## Pre-requisites

Running examples form this tutorial requires:

* Arhat SDK with valid trial or production license
* supported OS: Windows or Linux
* C++ compiler and development tools
* [Go](https://golang.org/) development tools version 1.14 or higher
* for CUDA backend: NVIDIA [CUDA Toolkit](https://developer.nvidia.com/cuda-zone)
version 10 or higher with the compatible [cuDNN](https://developer.nvidia.com/cudnn) library
* for CUDA backend: [CUB](https://github.com/NVlabs/cub) library version 1.8.0
* for CUDA backend: NVIDIA GPU device supporting CUDA
* for oneDNN backend: [oneDNN](https://github.com/oneapi-src/oneDNN) Deep Neural Network Library
version 1.5 or higher
* for oneDNN backend: Intel CPU or GPU device supporting oneDNN
* for ingesting user-supplied data sets: 
[Arhat NNEF Tools](https://github.com/fragata-ai/arhat-nnef)
* for extracting weights from torchvision model files:
[PyTorch](https://github.com/pytorch/pytorch) version 1.5.0 or higher

Examples in this tutorial assume Linux as OS. Development on Windows is conceptually similar.

## Supported target platforms

The following target platforms are currently supported:

`cuda`

> [CUDA](https://developer.nvidia.com/cuda-zone) and 
[cuDNN](https://developer.nvidia.com/cudnn) for NVIDIA GPU devices

`onednn`

> [oneDNN](https://github.com/oneapi-src/oneDNN) Deep Neural Network Library
for Intel CPU and GPU devices 

## Model nomenclature

Currently we provide a collection of models implementing a wide range of
image classification algorithms. These models have been borrowed from
the PyTorch [torchvision](https://pytorch.org/docs/stable/torchvision/index.html)
package. We have ported the original Python model descriptions to Go and
extracted weights from the original PyTorch model parameter files.

The following model faimilies are included in this collection:

* [AlexNet](https://arxiv.org/abs/1404.5997)
* [VGG](https://arxiv.org/abs/1409.1556)
* [ResNet](https://arxiv.org/abs/1512.03385)
* [SqueezeNet](https://arxiv.org/abs/1602.07360)
* [DenseNet](https://arxiv.org/abs/1608.06993)
* [Inception](https://arxiv.org/abs/1512.00567) v3
* [GoogLeNet](https://arxiv.org/abs/1409.4842)
* [ShuffleNet](https://arxiv.org/abs/1807.11164) v2
* [MobileNet](https://arxiv.org/abs/1801.04381) v2
* [ResNeXt](https://arxiv.org/abs/1611.05431)
* [Wide ResNet](https://arxiv.org/pdf/1605.07146.pdf)
* [MNASNet](https://arxiv.org/abs/1807.11626)

Some model families include multiple variants; each variant has
a unique name used to identify the respective model during target code
generation. The following model variants are available:

### AlexNet

`alexnet`

>AlexNet model architecture from the ["One weird trick..."](https://arxiv.org/abs/1404.5997>) paper. 

### VGG

`vgg11`

>VGG 11-layer model (configuration "A") from the
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf) paper.

`vgg11_bn`

>VGG 11-layer model (configuration "A") with batch normalization from the
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf) paper.

`vgg13`

>VGG 13-layer model (configuration "B") from the
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf) paper.

`vgg13_bn`

>VGG 13-layer model (configuration "B") from the
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf) paper.

`vgg16`

>VGG 16-layer model (configuration "D") from the
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf) paper.

`vgg16_bn`

>VGG 16-layer model (configuration "D") with batch normalization from the
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf) paper.

`vgg19`

>VGG 19-layer model (configuration "E") from the
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf) paper.

`vgg19_bn`

>VGG 19-layer model (configuration "E") with batch normalization from the
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf) paper.

### ResNet

`resnet18`

>ResNet-18 model from the
["Deep Residual Learning for Image Recognition"](https://arxiv.org/pdf/1512.03385.pdf) paper.

`resnet34`

>ResNet-34 model from the
["Deep Residual Learning for Image Recognition"](https://arxiv.org/pdf/1512.03385.pdf) paper.

`resnet50`

>ResNet-50 model from the
["Deep Residual Learning for Image Recognition"](https://arxiv.org/pdf/1512.03385.pdf) paper.

`resnet101`

>ResNet-101 model from the
["Deep Residual Learning for Image Recognition"](https://arxiv.org/pdf/1512.03385.pdf) paper.

`resnet152`

>ResNet-152 model from
["Deep Residual Learning for Image Recognition"](https://arxiv.org/pdf/1512.03385.pdf) paper.

### SqueezeNet

`squeezenet1_0`

>SqueezeNet model architecture from the ["SqueezeNet: AlexNet-level
accuracy with 50x fewer parameters and <0.5MB model size"](https://arxiv.org/abs/1602.07360) paper.

`squeezenet1_1`

>SqueezeNet 1.1 model from the 
[official SqueezeNet repo](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1).
SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
than SqueezeNet 1.0, without sacrificing accuracy.

### DenseNet

`densenet121`

>Densenet-121 model from the
[Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf) paper.

`densenet161`

>Densenet-161 model from the
[Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf) paper.

`densenet169`

>Densenet-169 model from the
["Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf) paper.

`densenet201`

>Densenet-201 model from the
["Densely Connected Convolutional Networks"](https://arxiv.org/pdf/1608.06993.pdf) paper.

### Inception v3

`inception_v3`

>Inception v3 model architecture from the
["Rethinking the Inception Architecture for Computer Vision"](http://arxiv.org/abs/1512.00567>) paper.

### GoogLeNet

`googlenet`

>GoogLeNet (Inception v1) model architecture from the
["Going Deeper with Convolutions"](http://arxiv.org/abs/1409.4842) paper. 

### ShuffleNet v2

`shufflenet_v2_x0_5`

>ShuffleNetV2 with 0.5x output channels, as described in the
["ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design]
(https://arxiv.org/abs/1807.11164) paper.

`shufflenet_v2_x1_0`

>ShuffleNetV2 with 1.0x output channels, as described in the
["ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"]
(https://arxiv.org/abs/1807.11164) paper.

`shufflenet_v2_x1_5`

>ShuffleNetV2 with 1.5x output channels, as described in the
["ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"]
(https://arxiv.org/abs/1807.11164) paper. Pre-trained parameters are currently
not available for this variant.

`shufflenet_v2_x2_0`

>ShuffleNetV2 with 2.0x output channels, as described in the
["ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"]
(https://arxiv.org/abs/1807.11164) paper. Pre-trained parameters are currently
not available for this variant.

### MobileNet v2

`mobilenet_v2`

>MobileNetV2 architecture from the
["MobileNetV2: Inverted Residuals and Linear Bottlenecks"](https://arxiv.org/abs/1801.04381) paper.

### ResNeXt

`resnext50_32x4d`

>ResNeXt-50 32x4d model from the
["Aggregated Residual Transformation for Deep Neural Networks"](https://arxiv.org/pdf/1611.05431.pdf) paper.

`resnext101_32x8d`

>ResNeXt-101 32x8d model from the
["Aggregated Residual Transformation for Deep Neural Networks"](https://arxiv.org/pdf/1611.05431.pdf) paper.

### Wide ResNet

`wide_resnet50_2`

>Wide ResNet-50-2 model from the
["Wide Residual Networks"](https://arxiv.org/pdf/1605.07146.pdf) paper.
The model is the same as ResNet except for the bottleneck number of channels
which is twice larger in every block. The number of channels in outer 1x1
convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
channels, and in Wide ResNet-50-2 has 2048-1024-2048.

`wide_resnet101_2`

>Wide ResNet-101-2 model from the
["Wide Residual Networks"](https://arxiv.org/pdf/1605.07146.pdf) paper.
The model is the same as ResNet except for the bottleneck number of channels
which is twice larger in every block. The number of channels in outer 1x1
convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
channels, and in Wide ResNet-50-2 has 2048-1024-2048.

### MNASNet

`mnasnet0_5`

>MNASNet with depth multiplier of 0.5 from the
["MnasNet: Platform-Aware Neural Architecture Search for Mobile]
(https://arxiv.org/pdf/1807.11626.pdf) paper.

`mnasnet0_75`

>MNASNet with depth multiplier of 0.75 from the
["MnasNet: Platform-Aware Neural Architecture Search for Mobile]
(https://arxiv.org/pdf/1807.11626.pdf) paper. Pre-trained parameters are currently
not available for this variant.

`mnasnet1_0`

>MNASNet with depth multiplier of 1.0 from the
["MnasNet: Platform-Aware Neural Architecture Search for Mobile]
(https://arxiv.org/pdf/1807.11626.pdf) paper.

`mnasnet1_3`

>MNASNet with depth multiplier of 1.3 from the
["MnasNet: Platform-Aware Neural Architecture Search for Mobile]
(https://arxiv.org/pdf/1807.11626.pdf) paper. Pre-trained parameters are currently
not available for this variant.

## Directory structure overview

**TODO**: Continue from this point

