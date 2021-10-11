# adjoint-project.jl
# test adjoint consistency for SPECT projector/back-projector on very small case

using SPECTrecon: SPECTplan
using SPECTrecon: project!, backproject!
using SPECTrecon: project, backproject
using LinearMapsAA: LinearMapAA
using LinearAlgebra: dot
using Test: @test, @testset
using Random: seed!


@testset "adjoint-project-matrix" begin
    T = Float32
    nx = 8; ny = nx
    nz = 6
    nview = 7

    mumap = rand(T, nx, ny, nz)

    nx_psf = 3
    nz_psf = 3
    psfs = rand(T, nx_psf, nz_psf, ny, nview)
    psfs = psfs .+ mapslices(reverse, psfs, dims = [1, 2]) # symmetrize
    psfs = psfs .+ mapslices(transpose, psfs, dims = [1, 2]) # symmetrize
    psfs = psfs ./ mapslices(sum, psfs, dims = [1, 2])

    dy = T(4.7952)
    idim = (nx,ny,nz)
    odim = (nx,nz,nview)

    for interpmeth in (:one, :two)
        for mode in (:fast, :mem)
            forw = x -> project(x, mumap, psfs, dy; interpmeth, mode)
            back = y -> backproject(y, mumap, psfs, dy; interpmeth, mode)
            A = LinearMapAA(forw, back, (prod(odim),prod(idim)); T, idim, odim)
            @test Matrix(A)' ≈ Matrix(A')
        end
    end

end


@testset "proj!-adj-test-dot" begin
    T = Float64
    nx = 32; ny = nx
    nz = 20
    nview = 30

    mumap = rand(T, nx, ny, nz)

    nx_psf = 7
    nz_psf = 7
    psfs = rand(T, nx_psf, nz_psf, ny, nview)
    psfs = psfs .+ mapslices(reverse, psfs, dims = [1, 2])
    psfs = psfs .+ mapslices(transpose, psfs, dims = [1, 2]) # symmetrize
    psfs = psfs ./ mapslices(sum, psfs, dims = [1, 2])

    dy = T(4.7952)
    seed!(0)

    for interpmeth in (:one, :two)
        for mode in (:fast, :mem)
            plan = SPECTplan(mumap, psfs, dy; interpmeth, mode)
            x = randn(T, nx, ny, nz)
            y = randn(T, nx, nz, nview)
            output_x = similar(y)
            output_y = similar(x)
            project!(output_x, x, plan)
            backproject!(output_y, y, plan)
            @test isapprox(dot(y, output_x), dot(x, output_y); rtol = 1e-5)
        end
    end

    for interpmeth in (:one, :two)
        for mode in (:fast, :mem)
            plan = SPECTplan(mumap, psfs, dy; interpmeth, mode)
            x = randn(T, nx, ny, nz)
            y = randn(T, nx, nz, nview)
            output_x = similar(y)
            output_y = similar(x)
            backproject!(output_y, y, plan)
            project!(output_x, x, plan)
            @test isapprox(dot(y, output_x), dot(x, output_y); rtol = 1e-5)
        end
    end

    for interpmeth in (:one, :two)
        for mode in (:fast, :mem)
            x = randn(T, nx, ny, nz)
            y = randn(T, nx, nz, nview)
            output_x = project(x, mumap, psfs, dy; interpmeth, mode)
            output_y = backproject(y, mumap, psfs, dy; interpmeth, mode)
            @test isapprox(dot(y, output_x), dot(x, output_y); rtol = 1e-5)
        end
    end

end
