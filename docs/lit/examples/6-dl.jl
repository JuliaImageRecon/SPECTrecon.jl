#---------------------------------------------------------
# # [SPECTrecon deep learning use](@id 6-dl)
#---------------------------------------------------------

#=
This page describes how to end-to-end train unrolled deep learning algorithms
using the Julia package
[`SPECTrecon`](https://github.com/JeffFessler/SPECTrecon.jl).

This page was generated from a single Julia file:
[6-dl.jl](@__REPO_ROOT_URL__/6-dl.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](http://nbviewer.jupyter.org/) here:
#md # [`6-dl.ipynb`](@__NBVIEWER_ROOT_URL__/6-dl.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`6-dl.ipynb`](@__BINDER_ROOT_URL__/6-dl.ipynb).


# ### Setup

# Packages needed here.

using LinearAlgebra: norm, mul!
using SPECTrecon
using MIRTjim: jim, prompt
using Plots: default; default(markerstrokecolor=:auto)
using Zygote
using ZygoteRules: @adjoint
using Flux: Chain, Conv, SamePad, relu, params, unsqueeze
using Flux # seems to be needed for BSON load; why?
using NNlib # seems to be needed for BSON load; why?
using LinearMapsAA: LinearMapAA
using Distributions: Poisson
using BSON: @load, @save
using InteractiveUtils: versioninfo

# The following line is helpful when running this example.jl file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


#=
### Overview

Regularized expectation-maximization (reg-EM)
is a commonly used algorithm for performing SPECT image reconstruction.
This page considers regularizers of the form β/2 * ||x - u||^2,
where u is an auxiliary variable that often refers to the image denoised by a CNN.

### Data generation

Simulated data used in this page are identical to
[`SPECTrecon ML-EM`](https://jefffessler.github.io/SPECTrecon.jl/stable/examples/4-mlem/).
We repeat it again here for convenience.
=#

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

forw! = (y,x) -> project!(y, x, plan)
back! = (x,y) -> backproject!(x, y, plan)
idim = (nx,ny,nz)
odim = (nx,nz,nview)
A = LinearMapAA(forw!, back!, (prod(odim),prod(idim)); T, odim, idim)


# Generate noisy data

if !@isdefined(ynoisy) # generate (scaled) Poisson data
    ytrue = A * xtrue
    target_mean = 20 # aim for mean of 20 counts per ray
    scale = target_mean / average(ytrue)
    scatter_fraction = 0.1 # 10% uniform scatter for illustration
    scatter_mean = scatter_fraction * average(ytrue) # uniform for simplicity
    ynoisy = rand.(Poisson.(scale * (ytrue .+ scatter_mean))) / scale
end
jim(ynoisy, "$nview noisy projection views"; ncol=10)


# ### ML-EM algorithm
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


# Apply MLEM algorithm to simulated data
x0 = ones(T, nx, ny, nz) # initial uniform image

niter = 30

if !@isdefined(xhat1)
    xhat1 = copy(x0)
    mlem!(xhat1, ynoisy, scatter_mean, A; niter)
end;

# Define evaluation metric
nrmse(x) = round(100 * norm(mid3(x) - mid3(xtrue)) / norm(mid3(xtrue)); digits=1)
jim(mid3(xhat1), "MLEM NRMSE=$(nrmse(xhat1))%") # display ML-EM reconstructed image


# ### Implement a 3D CNN denoiser

cnn = Chain(Conv((3,3,3), 1 => 4, relu; stride = 1, pad = SamePad(), bias = true),
            Conv((3,3,3), 4 => 4, relu; stride = 1, pad = SamePad(), bias = true),
            Conv((3,3,3), 4 => 1; stride = 1, pad = SamePad(), bias = true))
# show how many parameters the CNN has
paramCount = sum([sum(length, params(layer)) for layer in cnn])

prompt()


#=
### Custom backpropagation

Forward and back-projection are linear operations
so their Jacobians are very simple
and there is no need to auto-differentiate through the system matrix
and that would be very computationally expensive.
Instead, we tell Flux.jl to use the customized Jacobian when doing backpropagation.
=#

projectb(x) = A * x
@adjoint projectb(x) = A * x, dy -> (A' * dy, )

backprojectb(y) = A' * y
@adjoint backprojectb(y) = A' * y, dx -> (A * dx, )


# ### Backpropagatable regularized EM algorithm
# First define a function for unsqueezing the data todo: explain more
function unsqueeze45(x)
    return unsqueeze(unsqueeze(x, 4), 5)
end

"""
    bregem(projectb, backprojectb, y, r, Asum, x, cnn, β; niter = 1)

Backpropagatable regularized EM reconstruction with CNN regularization
-`projectb`: backpropagatable forward projection
-`backprojectb`: backpropagatable backward projection
-`y`: projections
-`r`: scatters
-`Asum`: A' * 1
-`x`: current iterate
-`cnn`: the CNN model
-`β`: regularization parameter
-`niter`: number of iteration for inner EM
"""
function bregem(
    projectb::Function,
    backprojectb::Function,
    y::AbstractArray,
    r::AbstractArray,
    Asum::AbstractArray,
    x::AbstractArray,
    cnn::Chain,
    β::Real;
    niter::Int = 1,
)

    u = cnn(unsqueeze45(x))[:,:,:,1,1]
    Asumu = Asum - β * u
    Asumu2 = Asumu.^2
    T = eltype(x)
    for iter = 1:niter
        eterm = backprojectb((y ./ (projectb(x) + r)))
        eterm_beta = 4 * β * (x .* eterm)
        x = max.(0, T(1/2β) * (-Asumu + sqrt.(Asumu2 + eterm_beta)))
    end
    return x
end


# ### Loss function
# We set β = 1 and train 2 outer iterations.

β = 1
Asum = A' * ones(T, nx, nz, nview)
scatters = scatter_mean * ones(T, nx, nz, nview)
function loss(xrecon, xtrue)
    xiter1 = bregem(projectb, backprojectb, ynoisy, scatters,
                    Asum, xrecon, cnn, β; niter = 1)
    xiter2 = bregem(projectb, backprojectb, ynoisy, scatters,
                    Asum, xiter1, cnn, β; niter = 1)
    return sum(abs2, xiter2 - xtrue)
end


# Initial loss
@show loss(xhat1, xtrue)


# ### Train the CNN
# Uncomment the following code to train!
## using Printf
## nepoch = 200
## for e = 1:nepoch
##     @printf("epoch = %d, loss = %.2f\n", e, loss(xhat1, xtrue))
##     ps = Flux.params(cnn)
##     gs = gradient(ps) do
##         loss(xhat1, xtrue) # we start with the 30 iteration EM reconstruction
##     end
##     opt = ADAMW(0.002)
##     Flux.Optimise.update!(opt, ps, gs)
## end

# Uncomment to save your trained model.
## @save "../data/trained-cnn-example-6-dl.bson" cnn

@info pwd()

# Load the pretrained model.
@load "../../data/trained-cnn-example-6-dl.bson" cnn

xiter1 = bregem(projectb, backprojectb, ynoisy, scatters,
                Asum, xhat1, cnn, β; niter = 1)
xiter2 = bregem(projectb, backprojectb, ynoisy, scatters,
                Asum, xiter1, cnn, β; niter = 1)

clim = (0,2)
jim(
    jim(mid3(xtrue), "xtrue"; clim),
    jim(mid3(xhat1), "EM recon, NRMSE = $(nrmse(xhat1))%"; clim),
    jim(mid3(xiter1), "Iter 1, NRMSE = $(nrmse(xiter1))%"; clim),
    jim(mid3(xiter2), "Iter 2, NRMSE = $(nrmse(xiter2))%"; clim),
)


# ### Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
