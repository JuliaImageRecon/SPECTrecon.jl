# ml-os-em.jl
# test mlem and osem algorithm

using SPECTrecon: SPECTplan
using SPECTrecon: project!, backproject!
using SPECTrecon: project, backproject
using SPECTrecon: Ablock
using SPECTrecon: mlem, mlem!, osem, osem!
using LinearMapsAA: LinearMapAA
using LinearAlgebra: dot
using Distributions: Poisson
using Test: @test, @testset

@testset "ml-os-em" begin
    nx,ny,nz = 32,32,20
    T = Float32
    xtrue = zeros(T, nx,ny,nz)
    xtrue[(1nx÷4):(2nx÷3), 1ny÷5:(3ny÷5), 2nz÷6:(3nz÷6)] .= 1
    xtrue[(2nx÷5):(3nx÷5), 1ny÷5:(2ny÷5), 4nz÷6:(5nz÷6)] .= 2
    px = 5
    psf1 = psf_gauss( ; ny, px)
    nview = 24
    psfs = repeat(psf1, 1, 1, 1, nview)
    dy = 8 # transaxial pixel size in mm
    mumap = zeros(T, size(xtrue)) # zero μ-map just for illustration here
    plan = SPECTplan(mumap, psfs, dy; T)
    forw! = (y,x) -> project!(y, x, plan)
    back! = (x,y) -> backproject!(x, y, plan)
    idim = (nx,ny,nz)
    odim = (nx,nz,nview)
    A = LinearMapAA(forw!, back!, (prod(odim),prod(idim)); T, odim, idim)
    ytrue = A * xtrue
    target_mean = 20 # aim for mean of 20 counts per ray
    average(x) = sum(x) / length(x)
    scale = target_mean / average(ytrue)
    scatter_fraction = 0.1 # 10% uniform scatter for illustration
    scatter_mean = scatter_fraction * average(ytrue) # uniform for simplicity
    ynoisy = rand.(Poisson.(scale * (ytrue .+ scatter_mean))) / scale
    x0 = ones(T, nx, ny, nz) # initial uniform image
    nblocks = 4
    niter = 10
    Ab = Ablock(plan, nblocks)
    xhat1 = mlem(x0, ynoisy, scatter_mean, A; niter)
    xhat2 = copy(x0)
    mlem!(xhat2, ynoisy, scatter_mean, A; niter)
    @test xhat1 ≈ xhat2
    xhat3 = osem(x0, ynoisy, scatter_mean * ones(T, nx, nz, nview), Ab; niter)
    xhat4 = copy(x0)
    osem!(xhat4, ynoisy, scatter_mean * ones(T, nx, nz, nview), Ab; niter)
    @test xhat3 ≈ xhat4
end
