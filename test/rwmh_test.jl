using MonteCarloMarkovKernels
using StaticArrays
using Random
using LinearAlgebra
import Statistics.mean

zero2 = [0.0, 0.0]
μ1 = [-1.0,0.0]
Σ1 = [1.0 1.5 ; 1.5 3.0]
propSigma = [2.0 1.2 ; 1.2 3.0]

Szero2 = SVector{2, Float64}(zero2)
SpropSigma = SMatrix{2, 2, Float64}(propSigma)

## slightly more complicated, but much more efficient approach
## uses StaticArrays and scratch space
function makelogMVN(μ::SVector{d, Float64}, Σ::SMatrix{d, d, Float64}) where d
  invΣ = inv(Σ)
  lognc = - 0.5 * d * log(2 * π) - 0.5 * logdet(Σ)
  function lpi(x::SVector{d, Float64})
    v = x - μ
    return lognc - 0.5*dot(v, invΣ * v)
  end
  return lpi
end

logtarget = makelogMVN(SVector{2, Float64}(μ1), SMatrix{2, 2, Float64}(Σ1))
P_RWM = makeRWMKernel(logtarget, SpropSigma)

seed!(12345)
chain = simulateChain(P_RWM, Szero2, 2^15)

cc = MonteCarloMarkovKernels.cov(chain)

@test maximum(abs.(mean(chain) - μ1)) < 0.1
@test maximum(abs.(cc - Σ1)) < 0.2
@test 0.2 < P_RWM(:acceptanceRate) < 0.5

## test that only custom RNG is used
Random.seed!(1); v1 = rand(); Random.seed!(1)
P_RWM = makeRWMKernel(logtarget, SpropSigma, MersenneTwister(12345))
chain = simulateChain(P_RWM, Szero2, 256)
v2 = rand()
@test v1 == v2
