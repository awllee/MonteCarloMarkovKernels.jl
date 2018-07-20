using KernelDensity
import Compat.Statistics.std

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

## can call contour(kde(vs, [acceptanceRate]))
function kde(vs::Vector{Float64}, acceptanceRate::Float64 = 1.0)
  hDefault = _defaultBandwidth(vs)
  hAdjusted = hDefault * (2 / acceptanceRate - 1)^(0.2)
  left = minimum(vs) - 3 * hAdjusted
  right = maximum(vs) + 3 * hAdjusted
  xs = linspace(left, right, 512)
  ys = pdf(InterpKDE(KernelDensity.kde(vs; bandwidth = hAdjusted)), xs)
  return xs, ys
end

## can call contour(kde(xs, ys, [acceptanceRate]))
function kde(xs::Vector{Float64}, ys::Vector{Float64},
  acceptanceRate::Float64 = 1.0)
  hAdjusted = _defaultBandwidth(xs, ys) * (2 / acceptanceRate - 1)^(1/6)

  left = minimum(xs) - 3 * hAdjusted[1]
  right = maximum(xs) + 3 * hAdjusted[1]
  bottom = minimum(ys) - 3 * hAdjusted[2]
  top = maximum(ys) + 3 * hAdjusted[2]
  xOut = linspace(left, right, 128)
  yOut = linspace(bottom, top, 128)
  ikde = InterpKDE(KernelDensity.kde((xs,ys); bandwidth = Tuple(hAdjusted)))
  function de(x, y)
    return pdf(ikde, x, y)
  end
  return xOut, yOut, de
end
