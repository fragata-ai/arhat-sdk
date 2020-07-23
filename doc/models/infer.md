
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
* [SqueezeNet[](https://arxiv.org/abs/1602.07360)
* [DenseNet](https://arxiv.org/abs/1608.06993)
* [Inception](https://arxiv.org/abs/1512.00567) v3
* [GoogLeNet](https://arxiv.org/abs/1409.4842)
* [ShuffleNet](https://arxiv.org/abs/1807.11164) v2
* [MobileNet](https://arxiv.org/abs/1801.04381) v2
* [ResNeXt](https://arxiv.org/abs/1611.05431)
* Wide ResNet
* [MNASNet](https://arxiv.org/abs/1807.11626)

Some model families can include multiple variants; each variant has
a unique name used to identify the respective model during target code
generation. The following model variants are available:

### AlexNet

`alexnet`

>AlexNet model architecture from the ["One weird trick..."](https://arxiv.org/abs/1404.5997>) paper. 

### VGG

`vgg11`

>VGG 11-layer model (configuration "A") from
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf).

`vgg11_bn`

>VGG 11-layer model (configuration "A") with batch normalization from
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf)

`vgg13`

>VGG 13-layer model (configuration "B") from
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf)

'vgg13_bn`

>VGG 13-layer model (configuration "B") from
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf)

`vgg16`

>VGG 16-layer model (configuration "D") from
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf)

`vgg16_bn`

>VGG 16-layer model (configuration "D") with batch normalization from
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf)

`vgg19`

>VGG 19-layer model (configuration "E") from
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf)

`vgg19_bn`

>VGG 19-layer model (configuration "E") with batch normalization from
["Very Deep Convolutional Networks For Large-Scale Image Recognition"](https://arxiv.org/pdf/1409.1556.pdf)

