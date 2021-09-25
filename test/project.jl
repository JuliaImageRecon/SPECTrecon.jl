# project.jl
using Main.SPECTrecon
using Test: @test, @testset, @test_throws, @inferred
using MAT

@testset "project" begin
    T = Float32
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

    output_x = Main.SPECTrecon.project(x, mumap, psfs, nviews, dy; interpidx = 1)
    output_y = Main.SPECTrecon.backproject(y, mumap, psfs, nviews, dy; interpidx = 1)
    @test isapprox(vdot(y, output_x), vdot(x, output_y))

    output_x = Main.SPECTrecon.project(x, mumap, psfs, nviews, dy; interpidx = 2)
    output_y = Main.SPECTrecon.backproject(y, mumap, psfs, nviews, dy; interpidx = 2)
    @test isapprox(vdot(y, output_x), vdot(x, output_y))
end
