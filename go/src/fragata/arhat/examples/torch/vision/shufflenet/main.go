
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
        return fmt.Errorf("Usage: torch_shufflenet platform modelName [outDir]\n")
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
    case "shufflenet_v2_x0_5":
        model = ShuffleNetV2x05(batchSize, nil)
    case "shufflenet_v2_x1_0":
        model = ShuffleNetV2x10(batchSize, nil)
    case "shufflenet_v2_x1_5":
        model = ShuffleNetV2x15(batchSize, nil)
    case "shufflenet_v2_x2_0":
        model = ShuffleNetV2x20(batchSize, nil)
    default:
        err = fmt.Errorf("Invalid model name: %s", name)
    }
    return
}

