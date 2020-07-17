
package main

import (
    "fmt"
    "os"
    "fragata/arhat/examples/torch/vision/util"
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
    if len(args) < 2 || len(args) > 3 {
        return fmt.Errorf("Usage: torch_alexnet platform [outDir]\n")
    }
    platform := args[1]
    var outDir string
    if len(args) >= 3 {
        outDir = args[2]
    } else {
        outDir = defaultOutDir
    }
    model := BuildAlexNet(batchSize, numClasses)
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

