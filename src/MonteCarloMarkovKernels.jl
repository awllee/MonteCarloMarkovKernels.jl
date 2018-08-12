module MonteCarloMarkovKernels
using StaticArrays
using LinearAlgebra
using Random
import Statistics.mean

include("simulateChain.jl")
include("randomWalkMetropolis.jl")
include("adaptiveMetropolis.jl")
include("batchMeans.jl")
include("spectralVariance.jl")
include("visualize.jl")

export simulateChain!, simulateChain,
  simulateChainProgress!, simulateChainProgress,
  makeRWMKernel, makeAMKernel,
  estimateBM, estimateSV,
  kde

end
