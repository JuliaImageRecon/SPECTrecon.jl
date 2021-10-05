# adjoint-project.jl
# test adjoint consistency for SPECT projector/back-projector on very small case

using SPECTrecon: project, backproject
using SPECTrecon: project!, backproject!
using SPECTrecon: SPECTplan, Workarray
using LinearMapsAA: LinearMapAA
using Test: @test, @testset


#@testset "adjoint" begin
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

    dy = 4.7952

#   for interpidx = 1:2
    interpidx = 1
        plan = SPECTplan(mumap, psfs, dy; interpidx)
#       workarray = Workarray(plan.T, plan.imgsize, plan.pad_fft, plan.pad_rot)
        workarray = [Workarray(plan.T, plan.imgsize, plan.pad_fft, plan.pad_rot) for i in 1:plan.ncore] # allocate
#       workarray = [workarray]
        forw = x -> project(x, mumap, psfs, dy; interpidx)
        back = y -> backproject(y, mumap, psfs, dy; interpidx)
        forw! = (y,x) -> project!(y, x, plan, workarray)
        back! = (x,y) -> backproject!(x, y, plan, workarray)
        idim = (nx,ny,nz)
        odim = (nx,nz,nview)
        A0 = LinearMapAA(forw, back, (prod(odim),prod(idim)); T, idim, odim)
        A! = LinearMapAA(forw!, back!, (prod(odim),prod(idim)); T, idim, odim)
        x = 0 * rand(T, idim)
        y = Array{T}(undef, odim)
        @show extrema(forw(x))
        @show extrema(forw!(y,x)) # NaN values even for 0 input :(
#       @test forw!(y,x) ≈ forw(x)
#       @test Matrix(A0)' ≈ Matrix(A0')
#       @test Matrix(A!)' ≈ Matrix(A!')
#    end
#end
