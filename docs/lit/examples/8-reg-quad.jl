#=
# [Quadratic regularization](@id 8-reg-quad)

This page illustrates SPECT image reconstruction
with quadratic regularization
using the Julia package
[`SPECTrecon`](https://github.com/JuliaImageRecon/SPECTrecon.jl).
=#

#srcURL

# ## Setup

# Packages needed here.

using SPECTrecon
using LinearAlgebra: dot, norm
using MIRTjim: jim, prompt
using MIRT: diffl_map
using Optim: optimize, LBFGS, Fminbox
import Optim #: Options
using Plots: scatter, plot!, default; default(markerstrokecolor=:auto)

# The following line is helpful when running this example.jl file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

#=
## Overview

SPECT image reconstruction
is an ill-conditioned problem
and maximum-likelihood estimation
leads to noise amplification
as iterations increase.
Adding a regularizer can control the noise.
This example uses quadratic regularization
of finite differences between neighboring voxels.
=#


# ## Simulation data

nx,ny,nz = 64,64,50
T = Float32
T = Float64
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


# ## PSF

# Create a synthetic depth-dependent PSF for a single view
px = 11
psf1 = psf_gauss( ; ny, px)
jim(psf1, "PSF for each of $ny planes")


#=
In general the PSF can vary from view to view
due to non-circular detector orbits.
For simplicity, here we illustrate the case
where the PSF is the same for every view.
=#

nview = 60
psfs = repeat(psf1, 1, 1, 1, nview)
size(psfs)


# ## SPECT system model using `LinearMapAA`

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


# ## Noisy data
using Distributions: Poisson

if !@isdefined(ynoisy) # generate (scaled) Poisson data
    ytrue = A * xtrue
    target_mean = 20 # aim for mean of 20 counts per ray
    scale = target_mean / average(ytrue)
    scatter_fraction = 0.1 # 10% uniform scatter for illustration
    scatter_mean = scatter_fraction * average(ytrue) # uniform for simplicity
    background = scatter_mean * ones(T,nx,nz,nview)
    ynoisy = rand.(Poisson.(scale * (ytrue + background))) / scale
end
jim(ynoisy, "$nview noisy projection views")


# ## Regularizer

Δ = diffl_map(size(xtrue), 1:3)
dtmp = Δ * xtrue;
jim(dtmp; title = "Δ * xtrue", ncol=25)


# ## Regularized cost function and gradient

beta = 1 # todo

function cost(x)
    yb = A * x + background
    neg_like = sum(yb) - dot(ynoisy, log.(yb)) # assumes background > 0
    return neg_like + beta * norm(Δ * x)^2 / 2
end
function grad(x)
    yb = A * x + background
    neg_like_grad = A' * (1 .- ynoisy ./ yb)
    return neg_like_grad + beta * (Δ' * (Δ * x))
end
function grad!(g, x)
    copyto!(g, grad(x))
end

# Optim.jl wants `Vector` arguments:
cost_vec = x -> cost(reshape(x, size(xtrue)))
grad_vec = x -> vec(grad(reshape(x, size(xtrue))))
function grad_vec!(g, x)
    copyto!(g, grad_vec(x))
end
if false # tests
    cost(xtrue)
    grad(xtrue)
    cost_vec(vec(xtrue))
    grad_vec(vec(xtrue))
end


x0 = ones(T, nx, ny, nz) # initial uniform image

lower = zeros(T, size(xtrue))
upper = fill(T(100), size(xtrue)) # max(xtrue) = 2
alg = LBFGS(; m = 10)
alg = Fminbox(alg)
opt = Optim.Options(
 iterations = 10,
 store_trace = true,
 show_trace = false,
)
optimize(cost_vec, grad_vec!, vec(lower), vec(upper), vec(x0), alg, opt)

#=
throw()

niter = 30
if !@isdefined(xhat1)
end
size(xhat1)
=#
