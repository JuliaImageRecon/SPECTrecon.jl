# runtests.jl

using Test: @test, @testset, detect_ambiguities
using Main.SPECTrecon

include("rotate3.jl")
include("project.jl")

@testset "SPECTrecon" begin
    @test isempty(detect_ambiguities(Main.SPECTrecon))
end
