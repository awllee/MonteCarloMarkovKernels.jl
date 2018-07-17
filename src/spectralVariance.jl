function γn(xs::Vector{Float64}, s::Int64)
  n::Int64 = length(xs)
  sa::Int64 = abs(s)
  xbar::Float64 = mean(xs)
  acc::Float64 = 0.0
  for i in 1:n-sa
    @inbounds acc += (xs[i]-xbar)*(xs[i+sa]-xbar)
  end
  return acc/n
end

function estimateSV(xs::Vector{Float64}, b::Int64, ws::Vector{Float64})
  v::Float64 = γn(xs, 0)
  for s in 1:b
    v += 2*ws[s]*γn(xs, s)
  end
  return v
end

function _wsSimpleTruncation(b::Int64)
  return ones(Float64, b)
end

function _wsBlackmanTukey(b::Int64, a::Float64)
  ws::Vector{Float64} = Vector{Float64}(b)
  for k in 1:b
    ws[k] = 1 - 2*a + 2*a*cos(π*k/b)
  end
  return ws
end

function _wsTukeyHanning(b::Int64)
  return _wsBlackmanTukey(b, 0.25)
end

function _wsParzen(b::Int64, q::Int64)
  ws::Vector{Float64} = Vector{Float64}(undef, b)
  for k in 1:b
    ws[k] = 1 - (k/b)^q
  end
  return ws
end

function _wsModifiedBartlett(b::Int64)
  return _wsParzen(b, 1)
end

function estimateSV(xs::Vector{Float64}, b::Int64, name::Symbol)
  name == :ModifiedBartlett && return estimateSV(xs, b, _wsModifiedBartlett(b))
end

function estimateSV(xs::Vector{Float64}, name::Symbol)
  b::Int64 = floor(Int64, sqrt(length(xs)))
  name == :ModifiedBartlett && return estimateSV(xs, b, _wsModifiedBartlett(b))
end
