using StaticArrays
using Compat.LinearAlgebra
using Compat.Random

if VERSION.minor == 6 mul! = A_mul_B! end

## Makes an adaptive Metropolis kernel proposed by:
## Haario, H., Saksman, E. and Tamminen, J., 2001. An adaptive Metropolis
## algorithm. Bernoulli, 7(2), pp.223-242.
## Essentially provides an adaptive mechanism to make use of the optimal
## scaling results for random walk Metropolis on "Gaussian-like" targets
## described by:
## Roberts, G.O. and Rosenthal, J.S., 2001. Optimal scaling for various
## Metropolis--Hastings algorithms. Statistical Science, 16(4), pp.351-367.
function makeAMKernel(logTargetDensity::F, Σ::SMatrix{d, d, Float64},
  updateFrequency::Int64 = 1, ϵ::Float64 = 1.0) where {F<:Function, d}
  S::MMatrix{d, d, Float64} = Σ
  # A::MMatrix{d, d, Float64} = chol(Symmetric(S))'
  A::MMatrix{d, d, Float64} = mychol(S)

  scratchv::MVector{d, Float64} = MVector{d, Float64}()
  scratchz::MVector{d, Float64} = MVector{d, Float64}()
  prevx::MVector{d, Float64} = MVector{d, Float64}()
  ldprevx = Ref(-Inf)

  accepts = Ref(0)
  calls = Ref(0)
  covEstimate::MMatrix{d, d, Float64} = Σ
  meanEstimate::MVector{d, Float64} = zeros(MVector{d, Float64})
  @inline function retuneSigma()
    if covEstimate[1,1] == 0.0
      S .= Σ  * ϵ / calls.x
    else
      S .= 5.6644 / d * covEstimate
    end
    try
      # A .= chol(Symmetric(S))'
      A .= mychol(S)
      ## quick fix as chol on SMatrix doesn't throw not pos def exceptions
      any(isnan, A) && throw(DomainError())
    catch e
      S .= Σ  * ϵ / calls.x
      A .= chol(Symmetric(S))'
    end
  end
  @inline function P(x::SVector{d, Float64})
    calls.x += 1
    randn!(scratchv)
    mul!(scratchz, A, scratchv)
    z::SVector{d, Float64} = scratchz + x
    # scratchv .= A * scratchv
    # z::SVector{d, Float64} = x + scratchv
    if x == prevx
      lpi_x = ldprevx.x
    else
      lpi_x = logTargetDensity(x)
      prevx .= x
    end
    lpi_z = logTargetDensity(z)
    if -randexp() < lpi_z - lpi_x
      prevx .= z
      ldprevx.x = lpi_z
      accepts.x += 1
      rval = z
    else
      rval = x
    end
    t::Int64 = calls.x
    covEstimate .= (t-1)/t * (covEstimate +
      (rval - meanEstimate) * (rval - meanEstimate)' / t)
    meanEstimate .= (t-1)/t.*meanEstimate + rval/t
    mod(t, updateFrequency) == 0 && retuneSigma()
    return rval
  end
  @inline function P(s::Symbol)
    s == :acceptanceRate && return accepts.x / calls.x
    s == :meanEstimate && return meanEstimate
    s == :covEstimate && return covEstimate * calls.x /(calls.x-1)
  end
  return P
end

function makeAMKernel(logTargetDensity::F, d::Int64, updateFrequency::Int64 = 1,
  ϵ::Float64 = 1.0) where F<:Function
  Id = SMatrix{d, d, Float64}(Matrix(1.0I, d, d))
  return makeAMKernel(logTargetDensity::F, Id, updateFrequency, ϵ)
end
