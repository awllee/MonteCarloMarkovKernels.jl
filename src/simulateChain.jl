using ProgressMeter

function simulateChain!(chain::Vector{T}, P::F, x0::T) where {F<:Function, T}
  n::Int64 = length(chain)
  x::T = x0
  for i = 1:n
    x = P(x)
    @inbounds chain[i] = x
  end
end

function simulateChain(P::F, x0::T, n::Int64) where {F<:Function, T}
  chain::Vector{T} = Vector{T}(undef, n)
  simulateChain!(chain, P, x0)
  return chain
end

function simulateChainProgress!(chain::Vector{T}, P::F, x0::T) where
  {F<:Function, T}
  n::Int64 = length(chain)
  x::T = x0
  @showprogress 10 for i = 1:n
    x = P(x)
    @inbounds chain[i] = x
  end
  return chain
end

function simulateChainProgress(P::F, x0::T, n::Int64) where {F<:Function, T}
  chain::Vector{T} = Vector{T}(undef, n)
  simulateChainProgress!(chain, P, x0)
  return chain
end

function cov(xs::Vector{SVector{d, Float64}}) where d
  xbar = mean(xs)
  Q::SMatrix{d, d, Float64} = zeros(SMatrix{d, d, Float64})
  for i = 1:length(xs)
    @inbounds Q += (xs[i] - xbar)*(xs[i] - xbar)'
  end
  return Q / (length(xs)-1)
end
