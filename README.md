
# Arhat SDK Preview

This repository contains a preview of documentation, tutorials, and sample code for Arhat SDK.

## About Arhat

Arhat is a deep learning software package that converts neural network descriptions into 
lean standalone executable code which can be directly deployed on various target platforms 
as part of user applications. This approach can provide significant benefits when 
used in embedded, cloud, and supercomputer applications because of simple and 
straightforward deployment not requiring installation of any conventional deep learning 
framework on the target platform. The standalone C++ code generated from the models 
can be directly integrated with a user application or any domain-specific software stack.

## Contents

This repository is work in progress and more materials will be added as soon
as they become available.

The repository currently contains the following resources:

* [Operator catalog](doc/op/catalog.md)
* [Models ported from NNEF model zoo](go/src/fragata/arhat/examples/zoo/nnef)
* [Models ported from torchvision](go/src/fragata/arhat/examples/torch/vision)
* [Tutorial: Getting started with Arhat models](doc/models/infer.md)

## License

We release the materials in this repository under an open source 
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) License.

