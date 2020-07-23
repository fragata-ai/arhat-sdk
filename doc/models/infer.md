
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
["ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture 
Design"](https://arxiv.org/abs/1807.11164) paper.

`shufflenet_v2_x1_0`

>ShuffleNetV2 with 1.0x output channels, as described in the
["ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture 
Design"](https://arxiv.org/abs/1807.11164) paper.

`shufflenet_v2_x1_5`

>ShuffleNetV2 with 1.5x output channels, as described in the
["ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture 
Design"](https://arxiv.org/abs/1807.11164) paper. Pre-trained parameters are currently
not available for this variant.

`shufflenet_v2_x2_0`

>ShuffleNetV2 with 2.0x output channels, as described in the
["ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture 
Design"](https://arxiv.org/abs/1807.11164) paper. Pre-trained parameters are currently
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
["MnasNet: Platform-Aware Neural Architecture Search for 
Mobile"](https://arxiv.org/pdf/1807.11626.pdf) paper.

`mnasnet0_75`

>MNASNet with depth multiplier of 0.75 from the
["MnasNet: Platform-Aware Neural Architecture Search for 
Mobile"](https://arxiv.org/pdf/1807.11626.pdf) paper. Pre-trained parameters are currently
not available for this variant.

`mnasnet1_0`

>MNASNet with depth multiplier of 1.0 from the
["MnasNet: Platform-Aware Neural Architecture Search for 
Mobile](https://arxiv.org/pdf/1807.11626.pdf) paper.

`mnasnet1_3`

>MNASNet with depth multiplier of 1.3 from the
["MnasNet: Platform-Aware Neural Architecture Search for 
Mobile"](https://arxiv.org/pdf/1807.11626.pdf) paper. Pre-trained parameters are currently
not available for this variant.

## Top-level directory structure

Top-level directories of Arhat SDK distribution have the following structure

```
arhat-sdk
      doc         // documentation
      cpp         // C++ code
      data        // sample data sets
      go          // Go code
```

We assume that the distribution root directory is `arhat-sdk`. Relative path references 
mentioned in this document are always assumed to be resolved using this root directory
as the base. For convenience, we set en environment variable `$ARHAT` to the absolute
path of this directory, e.g.:

```
export ARHAT=<absolute-path>/arhat-sdk
```

where `<absolute-path>` is an absolute path of `arhat-sdk` parent directory.

## Building inference programs from pre-fabricated C++ source code

The C++ code branch has the following directory structure:

```
arhat-sdk
    cpp
        bin                       // executables (created by build process)
        lib                       // libraries (created by build process)
        test                      // tests
        src                       // source code
            models                // pre-fabricated C++ code for models
                cuda              // code for CUDA/cuDNN
                onednn            // code for oneDNN
            runtime               // runtime functionality common for all platforms
            runtime_cuda          // runtime functionality specific for CUDA/cuDNN 
            runtime_onednn        // runtime functionality specific for oneDNN 
        prj                       // shell scripts for compilation and linking
        vendor                    // third party components
            cub                   // CUB library
            onednn                // oneDNN library
                bin
                include
                lib 
                share
```

Third party components that currently include CUB and oneDNN libraries are not included
in the distribution. They shall be obtained from the respective sources and placed
in the respective subtrees of `cpp/vendor`.

To start building process, make `cpp/prj` your current directory:

```
cd $ARHAT/cpp/prj
``` 

Proceed with building the runtime libraries. These libraries represent very lean collections
of helper functions. Use the respective command line scripts:

```
./runtime.bld
./runtime_cuda.bld
./runtime_onednn.bld
```

The respective libraries will be places in `cpp/lib`.

Then continue with building inference applications for selected models and target platforms.
The source code branches for have uniform structure for all platforms; below is the branch 
corresponding to CUDA:

```
arhat-sdk
    cpp
        src
            models
                cuda
                    torch
                        vision
                            alexnet
                            densenet121
                            densenet161
                            densenet169
                            densenet201
                            googlenet
                            inception_v3
                            mnasnet0_5
                            mnasnet1_0
                            mobilenet_v2
                            resnet101
                            resnet152
                            resnet18
                            resnet34
                            resnet50
                            shufflenet_v2_x0_5
                            shufflenet_v2_x1_0
                            squeezenet1_0
                            squeezenet1_1
                            vgg11
                            vgg11_bn
                            vgg13
                            vgg13_bn
                            vgg16
                            vgg16_bn
                            vgg19
                            vgg19_bn
                onednn
                    torch
                        vision
                            ...
```

Each leaf directory contains source code of the inference program for the a model
with the respective identifier. To build these programs, use scripts `cuda_torch_vision.bld` and 
`onednn_torch_vision.bld` for CUDA and oneDNN platforms respectively. 
Pass the model identifier as the sole script argument.
The scripts will produce executable files `cpp/lib` named `<platform>_torch_<model>` where
`<platform>` is a platform identifier and `model` is a model identifier.

For example, to build ResNet50 inference program for both platforms, use commands

```
./cuda_torch_vision.bld resnet50 
./onednn_torch_vision.bld resnet50 
``` 

This will produce files `cuda_torch_resnet50` and `onednn_torch_50` in `cpp/lib`.

Then, before running the inference programs, we shall obtain the model parameters
and input data sets.

## Obtaining the pre-trained model parameters

Because of their significant size, the pre-trained parameters are not included in
the Arhat SDK distribution. They can be obtained using two different methods.

In both cases, make `cpp/test` your current directory:

```
cd $ARHAT/cpp/test
```

The first method downloads compressed models from the cloud, uncompresses them
and places in the distinct subdirectories in `cpp/test`. To use this method,
run the shell script `load_param.sh` located in `cpp/test`:

```
./load_param.sh
```

You can also extract fragments from this script to download only parameters for
selected models.

The second method extracts model parameters from the original PyTorch parameter files
also located in the cloud. Using this method requires PyTorch installed. Run the shell
script `extract_param.sh` located in `cpp/test`:

```
./extract_param.sh
```

This script will launch the Python program `extract_param.py` that will download
the original PyTorch parameter files, extract parameters and convert them to
the NNEF data format; the results will be places in the same way as with
the first method. You can also edit `extract_param.py` to extract only parameters
for selected models.

On both cases, you shall get the following directory structure:

```
arhat-sdk
    cpp
        test
            alexnet
            densenet121
            densenet161
            densenet169
            densenet201
            googlenet
            inception_v3_google
            mnasnet0_5
            mnasnet1_0
            mobilenet_v2
            resnet
            resnet101
            resnet152
            resnet18
            resnet34
            resnet50
            resnext101_32x8d
            resnext50_32x4d
            shufflenetv2_x0.5
            shufflenetv2_x1.0
            squeezenet1_0
            squeezenet1_1
            vgg11
            vgg11_bn
            vgg13
            vgg13_bn
            vgg16
            vgg16_bn
            vgg19
            vgg19_bn
            wide_resnet101_2
            wide_resnet50_2
```

Note that in some cases the directory names differ from the corresponding model identifiers
(we intend to fix this in the future).

## Obtaining the input data sets

We will start with pre-fabricated data sets available at `data`. These sets have been created
from 10 arbitrarily selected husky images from the ImagaNet data set; these images can be
found in `data/husky`. The original images have been pre-processed in a conventional way,
that is, downsampled to the fixed size, RGB components scaled into a range of [0, 1] and then normalized using 
mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. Results have been stored
in NNEF data format repsesenting a single batch of 10 NCHW images.

There are two data sets downsamples to different sizes, namely 224 x 224 and 299 x 299.
The size 299 is shall be used for `inception_v3` input; all other models require 224.

Copy the pre-fabricated models to the `cpp/test' directory:

```
mkdir input
cp $ARHAT/data/husky224.dat ./input
cp $ARHAT/data/husky299.dat ./input
```

We are ready to run the inference programs now. Later we will explain how to build
input data sets from the user-supplied images.

## Running the inference programs



**TODO**: Continue from this point

