## Makes a random walk Metropolis kernel
function makeRWMKernel(logTargetDensity::F,
  Σ::SMatrix{d, d, Float64}, rng::RNG=Random.GLOBAL_RNG) where
  {F<:Function, d, RNG<:AbstractRNG}

  A::SMatrix{d, d, Float64} = cholesky(Σ).L
  scratchv::MVector{d, Float64} = MVector{d, Float64}(undef)
  prevx::MVector{d, Float64} = MVector{d, Float64}(undef)
  ldprevx = Ref(-Inf)
  accepts = Ref(0)
  calls = Ref(0)

  function P(x::SVector{d, Float64})
    calls.x += 1
    randn!(rng, scratchv)
    scratchv .= A * scratchv
    z::SVector{d, Float64} = x + scratchv
    if x == prevx
      lpi_x = ldprevx.x
    else
      lpi_x = logTargetDensity(x)
      prevx .= x
    end
    lpi_z = logTargetDensity(z)
    if -randexp(rng) < lpi_z - lpi_x
      prevx .= z
      ldprevx.x = lpi_z
      accepts.x += 1
      rval = z
    else
      rval = x
    end
    return rval
  end

  function P(s::Symbol)
    s == :acceptanceRate && return accepts.x / calls.x
  end
  
  return P
end
