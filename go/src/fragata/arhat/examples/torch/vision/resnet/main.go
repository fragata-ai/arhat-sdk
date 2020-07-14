
package main

import (
    "fmt"
    "os"
    "fragata/arhat/examples/torch/util"
    "fragata/arhat/front/models"
)

var (
    // Configurable parameters, change if necessary
    defaultOutDir = "./output"
    batchSize = 10
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
    if len(args) < 3 || len(args) > 4 {
        return fmt.Errorf("Usage: torch_resnet platform modelName [outDir]\n")
    }
    platform := args[1]
    modelName := args[2]
    var outDir string
    if len(args) >= 4 {
        outDir = args[3]
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
    engine, err := util.CreateEngine(platform)
    if err != nil {
        return err
    }
    err = model.InferOnBatch(engine.Graph(), -1)
    if err != nil {
        return err
    }
    err = engine.Emit(outDir)
    if err != nil {
        return err
    }
    return nil
}

func CreateModel(name string) (model *models.Model, err error) {
    switch name {
    case "resnet18":
        model = ResNet18(batchSize, nil)
    case "resnet34":
        model = ResNet34(batchSize, nil)
    case "resnet50":
        model = ResNet50(batchSize, nil)
    case "resnet101":
        model = ResNet101(batchSize, nil)
    case "resnet152":
        model = ResNet152(batchSize, nil)
    case "resnext50_32x4d":
        model = ResNeXt50_32x4d(batchSize, nil)
    case "resnext101_32x8d":
        model = ResNeXt101_32x8d(batchSize, nil)
    case "wide_resnet50_2":
        model = WideResNet50_2(batchSize, nil)
    case "wide_resnet101_2":
        model = WideResNet101_2(batchSize, nil)
    default:
        err = fmt.Errorf("Invalid model name: %s", name)
    }
    return
}

