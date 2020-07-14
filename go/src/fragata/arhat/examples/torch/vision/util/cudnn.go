
package util

import (
    emit "fragata/arhat/emit/cuda"
    "fragata/arhat/graph/api"
    graph "fragata/arhat/graph/cudnn"
)

var (
    // Configurable parameters, change if necessary
    // Too lazy to move them to the command line
    cudaVersion = 10200              // 10.2
    computeCapability = [2]int{7, 5} // 7.5
)

//
//    CudnnEngine
//

type CudnnEngine struct {
    emitter *emit.Emitter
    net *graph.Graph
}

// construction/destruction

func NewCudnnEngine() *CudnnEngine {
    e := new(CudnnEngine)
    e.emitter = emit.NewEmitter(cudaVersion, computeCapability)
    e.net = graph.NewGraph(e.emitter, e.emitter)
    return e
}

// interface

func(e *CudnnEngine) Graph() api.Graph {
    return e.net
}

func(e *CudnnEngine) Emit(outDir string) error {
    return e.emitter.Emit(e.net, outDir)
}

