# psf-gauss.jl

using SPECTrecon: psf_gauss
using Test: @test, @testset, @test_throws, @inferred


@testset "psf" begin
    psf = @inferred psf_gauss()
    @test psf isa Array{Float32,3}
end
