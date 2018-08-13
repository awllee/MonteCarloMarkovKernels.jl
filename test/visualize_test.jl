using MonteCarloMarkovKernels
using StaticArrays
import Statistics.mean
using Test

zero2 = [0.0, 0.0]
μ1 = [-1.0, 0.0]
Σ1 = [1.0 0.25 ; 0.25 3.0]

Szero2 = SVector{2, Float64}(zero2)

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

seed!(12345)

P_AM = makeAMKernel(logtarget, 2)
chain = simulateChain(P_AM, Szero2, 2^20)

@test mean(chain) ≈ μ1 atol = 0.01
@test MonteCarloMarkovKernels.cov(chain) ≈ Σ1 atol = 0.05

vs = (i->(x->x[i]).(chain)).(1:2)

d1(x) = 1/sqrt(2*π*Σ1[1,1])*exp(-1/2/Σ1[1,1]*(x-μ1[1])^2)
d2(x) = 1/sqrt(2*π*Σ1[2,2])*exp(-1/2/Σ1[2,2]*(x-μ1[2])^2)

xs, ys = kde(vs[1], P_AM(:acceptanceRate))
@test ys ≈ d1.(xs) atol=0.05

xs, ys = kde(vs[2], P_AM(:acceptanceRate))
@test ys ≈ d2.(xs) atol=0.05

xs, ys, f1 = kde(vs[1], vs[2], P_AM(:acceptanceRate))
M1 = broadcast(f1, xs, ys')
M2 = broadcast((x,y)->exp(logtarget((SVector{2,Float64}(x,y)))), xs, ys')
@test M1 ≈ M2 atol=0.05
