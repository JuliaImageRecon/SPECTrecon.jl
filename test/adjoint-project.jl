# adjoint.jl
# test adjoint on very small case

using SPECTrecon: project, backproject
using LinearMapsAA: LinearMapAA
using Test: @test, @testset


@testset "adjoint" begin
    T = Float32
    nx = 16
    ny = 16
    nz = 5
    nview = 7

    mumap = rand(T, nx, ny, nz)

    nx_psf = 5
    nz_psf = 3
    psfs = rand(T, nx_psf, nz_psf, ny, nview)
    psfs = psfs .+ mapslices(reverse, psfs, dims = [1, 2]) # symmetrize
    psfs = psfs ./ mapslices(sum, psfs, dims = [1, 2])

#   x = randn(T, nx, ny, nz)
    dy = 4.7952

    for interpidx = 1:2
        forw = x -> project(x, mumap, psfs, dy; interpidx)
        back = y -> backproject(y, mumap, psfs, dy; interpidx)
        idim = (nx,ny,nz)
        odim = (nx,nz,nview)
        A1 = LinearMapAA(forw, back, (prod(odim),prod(idim)); T, idim, odim)
        @test Matrix(A1)' ≈ Matrix(A1')
    end
end
