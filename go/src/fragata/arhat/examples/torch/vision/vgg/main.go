
package main

import (
    "fmt"
    "os"
    "fragata/arhat/examples/torch/vision/util"
    "fragata/arhat/front/models"
)

var (
    // Configurable parameters, change if necessary
    defaultOutDir = "./output"
    batchSize = 10
    numClasses = 1000
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
        return fmt.Errorf("Usage: torch_vgg platform modelName [outDir]\n")
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
    case "vgg11":
        model = Vgg11(batchSize, numClasses)
    case "vgg13":
        model = Vgg13(batchSize, numClasses)
    case "vgg16":
        model = Vgg16(batchSize, numClasses)
    case "vgg19":
        model = Vgg19(batchSize, numClasses)
    case "vgg11_bn":
        model = Vgg11Bn(batchSize, numClasses)
    case "vgg13_bn":
        model = Vgg13Bn(batchSize, numClasses)
    case "vgg16_bn":
        model = Vgg16Bn(batchSize, numClasses)
    case "vgg19_bn":
        model = Vgg19Bn(batchSize, numClasses)
    default:
        err = fmt.Errorf("Invalid model name: %s", name)
    }
    return
}

