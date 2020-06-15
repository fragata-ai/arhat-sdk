
package main

import (
    "fmt"
    "os"
    emit "fragata/arhat/emit/cuda"
    "fragata/arhat/front/models"
    graph "fragata/arhat/graph/cudnn"
)

var (
    // Configurable parameters, change if necessary
    cudaVersion = 10200              // 10.2
    computeCapability = [2]int{7, 5} // 7.5
    defaultOutDir = "./output"
)

func main() {
    err := Run()
    if err != nil {
        fmt.Fprintf(os.Stderr, "%s\n", err.Error())
        os.Exit(1)
    }
}

func Run() error {
    var err error
    args := os.Args
    if len(args) < 2 || len(args) > 3 {
        return fmt.Errorf("Usage: zoo_nnef modelName [outDir]\n")
    }
    modelName := args[1]
    var outDir string
    if len(args) >= 3 {
        outDir = args[2]
    } else {
        outDir = defaultOutDir
    }
    model, err := CreateModel(modelName)
    if err != nil {
        return err
    }
    err = model.Validate()
    if err != nil {
        return err
    }
    emitter := emit.NewEmitter(cudaVersion, computeCapability)
    net := graph.NewGraph(emitter, emitter)
    err = model.InferOnBatch(net, -1)
    if err != nil {
        return err
    }
    err = emitter.Emit(net, outDir)
    if err != nil {
        return err
    }
    return nil
}

func CreateModel(name string) (model *models.Model, err error) {
    switch name {
    case "alexnet_caffe":
        model = AlexnetCaffe()
    case "googlenet_caffe":
        model = GoogleNetCaffe()
    case "inception_v1_caffe2":
        model = InceptionV1Caffe2()
    case "inception_v2_caffe2":
        model = InceptionV2Caffe2()
    case "resnet_v1_18_onnx":
        model = Resnet18V1Onnx()
    case "resnet_v1_152_caffe":
        model = Resnet151V1Caffe()
    case "mobilenet_v1_10_caffe":
        model = MobilenetV1Caffe()
    case "mobilenet_v2_10_caffe":
        model = MobilenetV2Caffe()
    case "squeezenet_v10_caffe":
        model = SqueezeNetV10Caffe()
    case "squeezenet_v11_caffe":
        model = SqueezeNetV11Caffe()
    default:
        err = fmt.Errorf("Invalid model name: %s", name)
    }
    return
}

