function estimateBM(xs::Vector{Float64}, b::Int64)
  lxs::Int64 = length(xs)
  a = floor(Int64, lxs/b)
  @assert a > 1
  n::Int64 = a*b
  start::Int64 = length(xs) - n
  overallMean::Float64 = 0.0
  for i = 1:n
    overallMean += xs[start+i]
  end
  overallMean /= n
  acc::Float64 = 0.0
  for i = 1:a
    batchAcc::Float64 = 0.0
    batchStart::Int64 = start + (i-1)*b
    for j = 1:b
      batchAcc += xs[batchStart + j]
    end
    tmp::Float64 = batchAcc/b
    tmp -= overallMean
    acc += tmp * tmp
  end
  return b/(a-1)*acc
end

## Basic batch means estimation of the asymptotic variance
function estimateBM(xs::Vector{Float64})
  return estimateBM(xs, floor(Int64, sqrt(length(xs))))
end
