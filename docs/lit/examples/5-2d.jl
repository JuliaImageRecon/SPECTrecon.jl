#---------------------------------------------------------
# # [SPECTrecon 2D use](@id 5-2d)
#---------------------------------------------------------

# This page describes how to perform 2D SPECT forward and back-projection
# using the Julia package
# [`SPECTrecon`](https://github.com/JuliaImageRecon/SPECTrecon.jl).


# ### Setup

# Packages needed here.

using SPECTrecon
using MIRTjim: jim, prompt
using ImagePhantoms: shepp_logan, SheppLoganEmis
using LinearAlgebra: mul!
using LinearMapsAA: LinearMapAA
using Plots: plot, default; default(markerstrokecolor=:auto)

# The following line is helpful when running this example.jl file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


# ### Overview

# Real SPECT systems are inherently 3D imaging systems,
# but for the purpose of prototyping algorithms
# it can be useful to work with 2D simulations.

# Currently, "2D" here means a 3D array with `nz=1`,
# i.e., a single slice.
# The key to working with a single slice
# is that the package allows the PSFs
# to have rectangular support `px × pz`
# where `pz = 1`, i.e., no blur along the axial (z) direction.


# ### Example

# Start with a simple 2D digital phantom.

T = Float32
nx,ny,nz = 128,128,1
xtrue = T.(shepp_logan(nx, SheppLoganEmis()))
xtrue = reshape(xtrue, nx, ny, 1) # 3D array with nz=1
jim(xtrue, "xtrue: SheppLoganEmis with size $(size(xtrue))")


# ### PSF

# Create a synthetic depth-dependent PSF for a single view
px,pz = 11,1 # pz=1 is crucial for 2D work
psf1 = psf_gauss( ; ny, px, pz, fwhm_start = 1, fwhm_end = 4) # (px,pz,ny)
tmp = reshape(psf1, px, ny) / maximum(psf1) # (px,ny)
hx = (px-1)÷2
plot(-hx:hx, tmp[:,[1:9:end-10;end]], markershape=:o, label="",
    title = "Depth-dependent PSF profiles",
    xtick = [-hx, -2, 0, 2, hx], # (-1:1) .* ((px-1)÷2),
    ytick = [0; round.(tmp[hx+1,end] * [0.5,1], digits=2); 0.5; 1],
)
prompt()


# In general the PSF can vary from view to view
# due to non-circular detector orbits.
# For simplicity, here we illustrate the case
# where the PSF is the same for every view.

nview = 80
psfs = repeat(psf1, 1, 1, 1, nview)
size(psfs)


# ### Basic SPECT forward projection

# Here is a simple illustration
# of a SPECT forward projection operation.
# (This is a memory inefficient way to do it!)

dy = 4 # transaxial pixel size in mm
mumap = zeros(T, size(xtrue)) # μ-map just zero for illustration here
views = project(xtrue, mumap, psfs, dy) # [nx,1,nview]
sino = reshape(views, nx, nview)
size(sino)


# Display the calculated (i.e., simulated) projection views

jim(sino, "Sinogram")


# ### Basic SPECT back projection

# This illustrates an "unfiltered backprojection"
# that leads to a very blurry image
# (again, with a simple memory inefficient usage).

# First, back-project two "rays"
# to illustrate the depth-dependent PSF.
sino1 = zeros(T, nx, nview)
sino1[nx÷2, nview÷5] = 1
sino1[nx÷2, 1] = 1
sino1 = reshape(sino1, nx, nz, nview)
back1 = backproject(sino1, mumap, psfs, dy)
jim(back1, "Back-projection of two rays")


# Now back-project all the views of the phantom.

back = backproject(views, mumap, psfs, dy)
jim(back, "Back-projection of ytrue")


# ### Memory efficiency

# For iterative reconstruction,
# one must do forward and back-projection repeatedly.
# It is more efficient to pre-allocate work arrays
# for those operations,
# instead of repeatedly making system calls.

# Here we illustrate the memory efficient versions
# that are recommended for iterative SPECT reconstruction.

# First construction the SPECT plan.

#viewangle = (0:(nview-1)) * 2π # default
plan = SPECTplan(mumap, psfs, dy; T)

# Mutating version of forward projection:

tmp = Array{T}(undef, nx, nz, nview)
project!(tmp, xtrue, plan)
@assert tmp == views


# Mutating version of back-projection:

tmp = Array{T}(undef, nx, ny, nz)
backproject!(tmp, views, plan)
@assert tmp == back


# ### Using `LinearMapsAA`

# Calling `project!` and `backproject!` repeatedly
# leads to application-specific code.
# More general code uses the fact that SPECT projection and back-projection
# are linear operations,
# so we use `LinearMapAA` to define a "system matrix" for these operations.

forw! = (y,x) -> project!(y, x, plan)
back! = (x,y) -> backproject!(x, y, plan)
idim = (nx,ny,nz)
odim = (nx,nz,nview)
A = LinearMapAA(forw!, back!, (prod(odim),prod(idim)); T, odim, idim)

# Simple forward and back-projection:
@assert A * xtrue == views
@assert A' * views == back

# Mutating version:
tmp = Array{T}(undef, nx, nz, nview)
mul!(tmp, A, xtrue)
@assert tmp == views
tmp = Array{T}(undef, nx, ny, nz)
mul!(tmp, A', views)
@assert tmp == back


# ### Gram matrix impulse response

points = zeros(T, nx, ny, nz)
points[nx÷2,ny÷2,1] = 1
points[3nx÷4,ny÷4,1] = 1

impulse = A' * (A * points)
jim(impulse, "Impulse response of A'A")
