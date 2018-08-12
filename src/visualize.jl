using KernelDensity
import Statistics.std

function _defaultBandwidth(vs::Vector{Float64})
  return std(vs)*1.06*length(vs)^(-1/5)
end

function _defaultBandwidth(xs::Vector{Float64}, ys::Vector{Float64})
  @assert length(xs) == length(ys)
  len = length(xs)
  h1 = std(xs)*1.06*len^(-1/6)
  h2 = std(ys)*1.06*len^(-1/6)
  return [h1, h2]
end

## Compute adjusted kernel density estimates using an conservative approximation
## of the quantity proposed by:
## Sköld, M. and Roberts, G.O., 2003. Density estimation for the
## Metropolis–Hastings algorithm. Scandinavian Journal of Statistics, 30(4),
## pp.699-718.

## can call plot(kde(vs, [acceptanceRate]))
function kde(vs::Vector{Float64}, acceptanceRate::Float64, adjust::Float64)
  hDefault = _defaultBandwidth(vs)
  hAdjusted = hDefault * (2/acceptanceRate-1)^(0.2) * adjust
  left = minimum(vs) - 3 * hAdjusted
  right = maximum(vs) + 3 * hAdjusted
  xs = range(left, stop=right, length=512)
  ys = pdf(InterpKDE(KernelDensity.kde(vs; bandwidth = hAdjusted)), xs)
  return xs, ys
end

function kde(vs::Vector{Float64}, acceptanceRate::Float64 = 1.0)
  return kde(vs, acceptanceRate, 1.0)
end

## can call contour(kde(xs, ys, [acceptanceRate]))
function kde(xs::Vector{Float64}, ys::Vector{Float64},
  acceptanceRate::Float64, adjust::Float64)
  hAdjusted = _defaultBandwidth(xs, ys) * (2/acceptanceRate-1)^(1/6) * adjust

  left = minimum(xs) - 3 * hAdjusted[1]
  right = maximum(xs) + 3 * hAdjusted[1]
  bottom = minimum(ys) - 3 * hAdjusted[2]
  top = maximum(ys) + 3 * hAdjusted[2]
  xOut = range(left, stop=right, length=128)
  yOut = range(bottom, stop=top, length=128)
  ikde = InterpKDE(KernelDensity.kde((xs,ys); bandwidth = Tuple(hAdjusted)))
  function de(x, y)
    return pdf(ikde, x, y)
  end
  return xOut, yOut, de
end

function kde(xs::Vector{Float64}, ys::Vector{Float64},
  acceptanceRate::Float64 = 1.0)
  return kde(xs, ys, acceptanceRate, 1.0)
end
