__precompile__()

module MonteCarloMarkovKernels
using StaticArrays

using Compat.LinearAlgebra
using Compat.Random
import Compat.Statistics.mean

import Compat: undef, UndefInitializer
if VERSION.minor < 7
  MVector{d, Float64}(::UndefInitializer) where d = MVector{d, Float64}()
  mul! = A_mul_B!
end

if VERSION.minor == 7
  function mychol(A)
    return cholesky(A).L
  end
else
  function mychol(A)
    return chol(Symmetric(A))'
  end
end

include("simulateChain.jl")
include("randomWalkMetropolis.jl")
include("adaptiveMetropolis.jl")
include("batchMeans.jl")
include("spectralVariance.jl")
include("visualize.jl")

export simulateChain!, simulateChainProgress!,
  makeRWMKernel, makeAMKernel,
  estimateBM, estimateSV,
  kde

end
