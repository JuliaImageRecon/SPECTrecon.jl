#---------------------------------------------------------
# # [SPECTrecon ML-EM](@id 4-mlem)
#---------------------------------------------------------

# This page illustrates ML-EM reconstruction with the Julia package
# [`SPECTrecon`](https://github.com/JeffFessler/SPECTrecon.jl).

# ### Setup

# Packages needed here.

using SPECTrecon
using MIRTjim: jim, prompt
using Plots: scatter, plot!, default; default(markerstrokecolor=:auto)

# The following line is helpful when running this example.jl file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

# ### Overview

# Maximum-likelihood expectation-maximization (ML-EM)
# is a classic algorithm for performing SPECT image reconstruction.


# ### Simulation data

nx,ny,nz = 64,64,50
T = Float32
xtrue = zeros(T, nx,ny,nz)
xtrue[(1nx÷4):(2nx÷3), 1ny÷5:(3ny÷5), 2nz÷6:(3nz÷6)] .= 1
xtrue[(2nx÷5):(3nx÷5), 1ny÷5:(2ny÷5), 4nz÷6:(5nz÷6)] .= 2

average(x) = sum(x) / length(x)
function mid3(x::AbstractArray{T,3}) where {T}
    (nx,ny,nz) = size(x)
    xy = x[:,:,ceil(Int, nz÷2)]
    xz = x[:,ceil(Int,end/2),:]
    zy = x[ceil(Int, nx÷2),:,:]'
    return [xy xz; zy fill(average(xy), nz, nz)]
end
jim(mid3(xtrue), "Middle slices of xtrue")


# ### PSF

# Create a synthetic depth-dependent PSF for a single view
px = 11
psf1 = psf_gauss( ; ny, px)
jim(psf1, "PSF for each of $ny planes")


# In general the PSF can vary from view to view
# due to non-circular detector orbits.
# For simplicity, here we illustrate the case
# where the PSF is the same for every view.

nview = 60
psfs = repeat(psf1, 1, 1, 1, nview)
size(psfs)


# ### SPECT system model using `LinearMapAA`

dy = 8 # transaxial pixel size in mm
mumap = zeros(T, size(xtrue)) # zero μ-map just for illustration here
plan = SPECTplan(mumap, psfs, dy; T)

using LinearMapsAA: LinearMapAA
using LinearAlgebra: mul!
forw! = (y,x) -> project!(y, x, plan)
back! = (x,y) -> backproject!(x, y, plan)
idim = (nx,ny,nz)
odim = (nx,nz,nview)
A = LinearMapAA(forw!, back!, (prod(odim),prod(idim)); T, odim, idim)


# ### Basic Expectation-Maximization (EM) algorithm

# Noisy data
using Distributions: Poisson

if !@isdefined(ynoisy) # generate (scaled) Poisson data
    ytrue = A * xtrue
    target_mean = 20 # aim for mean of 20 counts per ray
    scale = target_mean / average(ytrue)
    scatter_fraction = 0.1 # 10% uniform scatter for illustration
    scatter_mean = scatter_fraction * average(ytrue) # uniform for simplicity
    ynoisy = rand.(Poisson.(scale * (ytrue .+ scatter_mean))) / scale
end
jim(ynoisy, "$nview noisy projection views")


# ### ML-EM algorithm - basic version

# This basic ML-EM version uses the linear map, but it is still allocating.

function mlem(x0, ynoisy, background, A; niter::Int = 20)
    all(>(0), background) || throw("need background > 0")
    x = copy(x0)
    asum = A' * ones(eltype(ynoisy), size(ynoisy))
    time0 = time()
    for iter = 1:niter
        @show iter, extrema(x), time() - time0
        ybar = A * x .+ background # forward model
        x .*= (A' * (ynoisy ./ ybar)) ./ asum # multiplicative update
    end
    return x
end


# This preferable ML-EM version modifies the input `x`,
# so no memory allocation is needed within the loop!

function mlem!(x, ynoisy, background, A; niter::Int = 20)
    all(>(0), background) || throw("need background > 0")
    asum = A' * ones(eltype(ynoisy), size(ynoisy)) # this allocates
    ybar = similar(ynoisy)
    yratio = similar(ynoisy)
    back = similar(x)
    time0 = time()
    for iter = 1:niter
        @show iter, extrema(x), time() - time0
        mul!(ybar, A, x)
        @. yratio = ynoisy / (ybar + background) # coalesce broadcast!
        mul!(back, A', yratio) # back = A' * (ynoisy / ybar)
        @. x *= back / asum # multiplicative update
    end
    return x
end


# Apply both versions of ML-EM to this simulated data

x0 = ones(T, nx, ny, nz) # initial uniform image

niter = 30
if !@isdefined(xhat1)
    xhat1 = mlem(x0, ynoisy, scatter_mean, A; niter)
end

if !@isdefined(xhat2)
    xhat2 = copy(x0)
    mlem!(xhat2, ynoisy, scatter_mean, A; niter)
end

@assert xhat1 ≈ xhat2

jim(mid3(xhat2), "ML-EM at $niter iterations")
