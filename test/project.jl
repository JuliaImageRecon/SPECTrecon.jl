# project.jl
using Main.SPECTrecon: project, backproject
using Test: @test, @testset, @test_throws, @inferred
using MAT
using LazyAlgebra: vdot 

@testset "project" begin
    T = Float32
    path = "./data/"
    file = matopen(path*"mumap208.mat")
    mumap = read(file, "mumap208")
    close(file)

    file = matopen(path*"psf_208.mat")
    psfs = read(file, "psf_208")
    close(file)
    dy = T(4.80)

    (nx, ny, nz) = size(mumap)
    nviews = size(psfs, 4)

    x = randn(T, nx, ny, nz)
    y = randn(T, nx, nz, nviews)

    output_x = project(x, mumap, psfs, nviews, dy; interpidx = 1)
    output_y = backproject(y, mumap, psfs, nviews, dy; interpidx = 1)
    @test isapprox(vdot(y, output_x), vdot(x, output_y))

    output_x = project(x, mumap, psfs, nviews, dy; interpidx = 2)
    output_y = backproject(y, mumap, psfs, nviews, dy; interpidx = 2)
    @test isapprox(vdot(y, output_x), vdot(x, output_y))
end
