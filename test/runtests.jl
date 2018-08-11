using MonteCarloMarkovKernels
using Test

import Random.seed!

@testset "AR1 test" begin
  @time include("ar1_test.jl")
end

@testset "RWM test" begin
  @time include("rwmh_test.jl")
end

@testset "AM test" begin
  @time include("am_test.jl")
end

@testset "Visualize test" begin
  @time include("visualize_test.jl")
end
