# runtests.jl

include("../src/SPECTrecon.jl")
using Test: @test, @testset, detect_ambiguities

include("helper.jl")
include("rotate3.jl")
include("fft_convolve.jl")
include("SPECTplan.jl")
include("project.jl")
include("backproject.jl")

@testset "SPECTrecon" begin
    @test isempty(detect_ambiguities(Main.SPECTrecon))
end
