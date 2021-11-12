# psf-gauss.jl

using SPECTrecon: psf_gauss
using Test: @test, @testset, @test_throws, @inferred


@testset "psf" begin
    psf = @inferred psf_gauss()
    @test psf isa Array{Float32,3}

    ny = 4
    px = 5
    pz = 3
    psf = @inferred psf_gauss(; ny, px, pz, fwhm = zeros(ny))
    tmp = zeros(px,pz)
    tmp[(end+1)÷2,(end+1)÷2] = 1 # Kronecker impulse
    tmp = repeat(tmp, 1, 1, ny)
    @test psf == tmp

    ny = 4
    px = 5
    pz = 3
    psf = @inferred psf_gauss(; ny, px, pz,
        fwhm_x = fill(Inf, ny),
        fwhm_z = zeros(ny),
    )
    tmp = zeros(px,pz)
    tmp[:,(end+1)÷2] .= 1/px # wide in x
    tmp = repeat(tmp, 1, 1, ny)
    @test psf ≈ tmp
end
