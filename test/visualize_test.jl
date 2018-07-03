using MonteCarloMarkovKernels
using StaticArrays
using StatsBase
using Compat.Test
using Plots
Plots.gr()

savefigures = false

zero2 = [0.0, 0.0]
μ1 = [-1.0, 0.0]
Σ1 = [1.0 0.25 ; 0.25 3.0]

Szero2 = SVector{2, Float64}(zero2)

function makelogMVN{d}(μ::SVector{d, Float64}, Σ::SMatrix{d, d, Float64})
  invΣ = inv(Σ)
  lognc = - 0.5 * d * log(2 * π) - 0.5 * logdet(Σ)
  function lpi(x::SVector{d, Float64})
    v = x - μ
    return lognc - 0.5*dot(v, invΣ * v)
  end
  return lpi
end

logtarget = makelogMVN(SVector{2, Float64}(μ1), SMatrix{2, 2, Float64}(Σ1))

niterations = 2^20
chain = Vector{SVector{2, Float64}}(niterations)

srand(12345)

P_AM = makeAMKernel(logtarget, 2)
simulateChain!(chain, P_AM, Szero2)

@test mean(chain) ≈ μ1 atol = 0.01
@test MonteCarloMarkovKernels.cov(chain) ≈ Σ1 atol = 0.05

!isinteractive() && (ENV["GKSwstype"] = "100")

vs = Vector{Vector{Float64}}(2)
for i = 1:2
  vs[i] = (x->x[i]).(chain)
end

d1(x) = 1/sqrt(2*π*Σ1[1,1])*exp(-1/2/Σ1[1,1]*(x-μ1[1])^2)
d2(x) = 1/sqrt(2*π*Σ1[2,2])*exp(-1/2/Σ1[2,2]*(x-μ1[2])^2)

xs, ys = kde(vs[1], P_AM(:acceptanceRate))
plot(xs, ys)
plot!(xs, d1.(xs), color="red")
savefigures && savefig("kde1.png")
@test ys ≈ d1.(xs) atol=0.05

xs, ys = kde(vs[2], P_AM(:acceptanceRate))
plot(xs, ys)
plot!(xs, d2.(xs), color="red")
savefigures && savefig("kde2.png")
@test ys ≈ d2.(xs) atol=0.05

xs, ys, f1 = kde(vs[1], vs[2], P_AM(:acceptanceRate))
contour(xs, ys, f1)
contour!(xs, ys, (x,y)->exp(logtarget((SVector{2,Float64}(x,y)))))
savefigures && savefig("kde12.png")

plot(autocor(vs[1]))
savefigures && savefig("acf1.png")
plot(autocor(vs[2]))
savefigures && savefig("acf2.png")
