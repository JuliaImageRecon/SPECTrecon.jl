# project.jl

using Main.SPECTrecon: project, backproject
using LazyAlgebra: vdot
using Test: @test, @testset, detect_ambiguities


@testset "proj-adj-test" begin
    T = Float32
    nx = 64
    ny = 64
    nz = 40
    nview = 60

    mumap = rand(T, nx, ny, nz)

    nx_psf = 19
    nz_psf = 19
    psfs = rand(T, nx_psf, nz_psf, ny, nview)
    psfs = psfs .+ mapslices(reverse, psfs, dims = [1, 2])
    psfs = psfs ./ mapslices(sum, psfs, dims = [1, 2])

    x = randn(T, nx, ny, nz)
    y = randn(T, nx, nz, nview)
    dy = T(4.7952)

    output_x = project(x, mumap, psfs, dy; interpidx = 1)
    output_y = backproject(y, mumap, psfs, dy; interpidx = 1)
    @test isapprox(vdot(y, output_x), vdot(x, output_y))

    output_x = project(x, mumap, psfs, dy; interpidx = 2)
    output_y = backproject(y, mumap, psfs, dy; interpidx = 2)
    @test isapprox(vdot(y, output_x), vdot(x, output_y))
end
