
package util

import (
    emit "fragata/arhat/emit/onednn"
    "fragata/arhat/graph/api"
    graph "fragata/arhat/graph/onednn"
)

//
//    OnednnEngine
//

type OnednnEngine struct {
    emitter *emit.Emitter
    net *graph.Graph
}

// construction/destruction

func NewOnednnEngine() *OnednnEngine {
    e := new(OnednnEngine)
    e.emitter = emit.NewEmitter()
    e.net = graph.NewGraph(e.emitter)
    return e
}

// interface

func(e *OnednnEngine) Graph() api.Graph {
    return e.net
}

func(e *OnednnEngine) Emit(outDir string) error {
    return e.emitter.Emit(e.net, outDir)
}

