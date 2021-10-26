#---------------------------------------------------------
# # [SPECTrecon overview](@id 1-overview)
#---------------------------------------------------------

# This page explains the Julia package
# [`SPECTrecon`](https://github.com/JeffFessler/SPECTrecon.jl).

# ### Setup

# Packages needed here.

using SPECTrecon
using MIRTjim: jim, prompt
using Plots: scatter, plot!, default; default(markerstrokecolor=:auto)
using Plots # @animate, gif

# The following line is helpful when running this example.jl file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

# ### Overview

# To perform SPECT image reconstruction,
# one must have a model for the imaging system
# encapsulated in a forward projector and back projector.

# Mathematically, we write the forward projection process in SPECT
# as "y = A * x" where A is a "system matrix"
# that models the physics of the imaging system
# (including depth-dependent collimator/detector response
# and attenuation)
# and "x" is the current guess of the emission image.

# However, in code we usually cannot literally store "A"
# as dense matrix because it is too large.
# A typical size in SPECT is that
# the image `x` is
# `nx × ny × nz = 128 × 128 × 100`
# and the array of projection views `y` is
# `nx × nz × nview = 128 × 100 × 120`.
# So the system matrix `A` has `1536000 × 1638400` elements
# which is far to many to store,
# even accounting for some sparsity.

# Instead, we write functions called forward projectors
# that calculate `A * x` "on the fly".

# Similarly, the operation `A' * y`
# is called "back projection",
# where `A'` denotes the transpose or "adjoint" of `A`.


# ### Example

# To illustrate forward and back projection,
# it is easiest to start with a simulation example
# using a digital phantom.
# The fancy way would be to use a 3D phantom from
# [ImagePhantoms](https://github.com/JuliaImageRecon/ImagePhantoms.jl),
# but instead we just use two simple cubes.

nx,ny,nz = 128,128,100
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
    return [xy xz; zy fill(average(x), nz, nz)]
end
jim(mid3(xtrue), "Middle slices of x")


# ### PSF
# Create a synthetic depth-dependent PSF for a single view

function fake_psf(nx::Int, nx_psf::Int; factor::Real=0.7)
    psf = zeros(T, nx_psf, nx_psf, nx)

    for iy in 1:nx # depth-dependent blur
        r = (-(nx_psf-1)÷2):((nx_psf-1)÷2)
        r2 = abs2.((r / nx_psf) * iy.^factor)
        tmp = @. exp(-(r2 + r2') / 2)
        psf[:,:,iy] = tmp / maximum(tmp)
    end
    return psf
end

nx_psf = 11
psf1 = fake_psf(nx, nx_psf)
jim(psf1, "PSF for each of $nx planes")


# In general the PSF can vary from view to view
# due to non-circular detector orbits.
# For simplicity, here we illustrate the case
# where the PSF is the same for every view.

nview = 120
psfs = repeat(psf1, 1, 1, 1, nview)


# Now plan the PSF modeling (see `3-psf.jl`)

plan = plan_psf(nx, nz, nx_psf)


# ### Basic SPECT forward projection

# Here is a simple illustration
# of a SPECT forward projection operation.
# (This is a memory inefficient way to do it!)

dy = 4 # transaxial pixel size in mm
mumap = zeros(T, size(xtrue)) # μ-map just zero for illustration here
views = project(xtrue, mumap, psfs, dy)

# Display the calculated (i.e., simulated) projection views

jim(views[:,:,1:4:end], "Every 4th of $nview projection views")


# ### Basic SPECT back projection

# This illustrates an "unfiltered backprojection"
# that leads to a very blurry image
# (again, with a simple memory inefficient usage).

# First, back-project a single "ray"
# to illustrate the depth-dependent PSF.
tmp = zeros(T, size(views))
tmp[nx÷2, nz÷2, nview÷5] = 1
tmp = backproject(tmp, mumap, psfs, dy)
jim(mid3(tmp), "Back-projection of one ray")


# Now back-projection all the views of the phantom

back = backproject(views, mumap, psfs, dy)
jim(mid3(back), "Back-projection of ytrue")


# ### Memory efficiency

# For iterative reconstruction,
# one must do forward and back-projection repeatedly.
# It is more efficient to pre-allocate work arrays
# for those operations,
# instead of repeatedly making system calls.

# Here we illustrate the memory efficient versions
# that are recommended for iterative SPECT reconstruction.

# First construction the SPECT plan

#viewangle = (0:(nview-1)) * 2π # default
plan = SPECTplan(mumap, psfs, dy; T)
summary(plan) # todo: concise "show"

# Mutating version of forward projection:

views = Array{T}(undef, nx, nz, nview)
project!(views, xtrue, plan)
jim(views[:,:,1:4:end], "Every 4th of $nview projection views")


# Mutating version of back-projection:

back = Array{T}(undef, nx, ny, nz)
backproject!(back, views, plan)
jim(mid3(back))


# ### Using `LinearMapAA`

# Calling `project!` and `backproject!` repeatedly
# leads to application-specific code.
# More general code uses the fact that SPECT projection and back-projection
# are linear operations,
# so we use `LinearMapAA` to define a "system matrix" for these operations.

using LinearMapsAA: LinearMapAA
forw! = (y,x) -> project!(y, x, plan)
back! = (x,y) -> backproject!(x, y, plan)
idim = (nx,ny,nz)
odim = (nx,nz,nview)
A = LinearMapAA(forw!, back!, (prod(odim),prod(idim)); T, odim, idim)

# Simple forward and back-projection:
ytrue = A * xtrue
back = A' * ytrue

# Mutating version:
using LinearAlgebra: mul!
mul!(ytrue, A, xtrue)
mul!(back, A', ytrue)
nothing


# ### Units

# The pixel dimensions `deltas` can (and should!) be values with units.

# Here is an example ... (todo)
#using UnitfulRecipes
#using Unitful: mm


# ### Projection view animation

anim = @animate for i in 1:nview
    ymax = maximum(ytrue)
    jim(ytrue[:,:,i],
        "SPECT projection view $i of $nview",
        clim = (0, ymax),
    )
end
gif(anim, "views.gif", fps = 8)
