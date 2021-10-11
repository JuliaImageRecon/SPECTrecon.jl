# runtests.jl

using SPECTrecon
using Test: @test, @testset, detect_ambiguities

include("helper.jl")
include("rotatez.jl")
include("adjoint-fftconv.jl")
include("adjoint-rotate.jl")
include("adjoint-project.jl")

@testset "SPECTrecon" begin
    @test isempty(detect_ambiguities(SPECTrecon))
end
