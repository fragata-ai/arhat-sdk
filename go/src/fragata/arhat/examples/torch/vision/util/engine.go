
package util

import (
    "fmt"
    "fragata/arhat/graph/api"
)

//
//    Engine
//

type Engine interface {
    Graph() api.Graph 
    Emit(outdir string) error
}

// factory

func CreateEngine(platform string) (engine Engine, err error) {
    switch platform {
    case "cudnn":
        engine = NewCudnnEngine()
    case "onednn":
        engine = NewOnednnEngine()
    default:
        err = fmt.Errorf("Unsupported platform: %s", platform)
    }
    return
}

