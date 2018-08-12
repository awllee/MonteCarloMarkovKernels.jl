using MonteCarloMarkovKernels
using Random
import Statistics: mean, var
using Test

seed!(12345)

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

chain = simulateChain(P, 0.0, 1024*1024)

@test mean(chain) ≈ c/(1-φ) rtol=0.1
@test var(chain) ≈ σ^2/(1-φ^2) rtol=0.1

@test estimateBM(chain) ≈ σ^2/(1-φ^2) * (1+φ)/(1-φ) rtol=0.5
@test estimateSV(chain, :ModifiedBartlett) ≈
  σ^2/(1-φ^2) * (1+φ)/(1-φ) rtol=0.5
