__precompile__()

module MonteCarloMarkovKernels

import Compat.undef
if VERSION.minor == 7
  import Statistics.mean
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
