# adjoint-project.jl
# test adjoint consistency for SPECT projector/back-projector on very small case

using SPECTrecon: project, backproject
using LinearMapsAA: LinearMapAA
using LinearAlgebra: dot
using Test: @test, @testset


@testset "adjoint-project-matrix" begin
    T = Float32
    nx = 16; ny = nx
    nz = 10
    nview = 7

    mumap = rand(T, nx, ny, nz)

    nx_psf = 5
    nz_psf = 5
    psfs = rand(T, nx_psf, nz_psf, ny, nview)
    psfs = psfs .+ mapslices(reverse, psfs, dims = [1, 2]) # symmetrize
    psfs = psfs .+ mapslices(transpose, psfs, dims = [1, 2]) # symmetrize
    psfs = psfs ./ mapslices(sum, psfs, dims = [1, 2])

    dy = T(4.7952)
    idim = (nx,ny,nz)
    odim = (nx,nz,nview)

    forw1 = x -> project(x, mumap, psfs, dy; interpmeth = :one)
    back1 = y -> backproject(y, mumap, psfs, dy; interpmeth = :one)
    A1 = LinearMapAA(forw1, back1, (prod(odim),prod(idim)); T, idim, odim)
    @test Matrix(A1)' ≈ Matrix(A1')

    forw2 = x -> project(x, mumap, psfs, dy; interpmeth = :two)
    back2 = y -> backproject(y, mumap, psfs, dy; interpmeth = :two)
    A2 = LinearMapAA(forw2, back2, (prod(odim),prod(idim)); T, idim, odim)
    @test Matrix(A2)' ≈ Matrix(A2')

end


@testset "proj-adj-test-dot" begin
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
    psfs = psfs .+ mapslices(transpose, psfs, dims = [1, 2]) # symmetrize
    psfs = psfs ./ mapslices(sum, psfs, dims = [1, 2])

    x = randn(T, nx, ny, nz)
    y = randn(T, nx, nz, nview)
    dy = T(4.7952)

    output_x = project(x, mumap, psfs, dy; interpmeth = :one)
    output_y = backproject(y, mumap, psfs, dy; interpmeth = :one)
    @test isapprox(dot(y, output_x), dot(x, output_y))

    output_x = project(x, mumap, psfs, dy; interpmeth = :two)
    output_y = backproject(y, mumap, psfs, dy; interpmeth = :two)
    @test isapprox(dot(y, output_x), dot(x, output_y))
end
