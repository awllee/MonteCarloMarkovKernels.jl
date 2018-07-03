using MonteCarloMarkovKernels
using Compat.Random
import Compat.undef
using Compat.Test

srand(12345)

function makeAR1Kernel(c::Float64, φ::Float64, σ::Float64)
  function P(x::Float64)
    return c + φ * x + σ * randn()
  end
  return P
end

const c = 2.32
const φ = 0.98
const σ = 2.3
P = makeAR1Kernel(c, φ, σ)

n = 1024*1024
chain = Vector{Float64}(undef, n)

simulateChain!(chain, P, 0.0)

@test mean(chain) ≈ c/(1-φ) rtol=0.1
@test var(chain) ≈ σ^2/(1-φ^2) rtol=0.1
# @test estimateAvar(chain) ≈ σ^2/(1-φ^2) * (1+φ)/(1-φ) rtol=0.1
@test estimateBM(chain) ≈ σ^2/(1-φ^2) * (1+φ)/(1-φ) rtol=0.5
@test estimateSV(chain, :ModifiedBartlett) ≈
  σ^2/(1-φ^2) * (1+φ)/(1-φ) rtol=0.5
