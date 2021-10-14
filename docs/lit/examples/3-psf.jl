#---------------------------------------------------------
# # [SPECTrecon PSF](@id 3-psf)
#---------------------------------------------------------

# This page explains the PSF portion of the Julia package
# [`SPECTrecon.jl`](https://github.com/JeffFessler/SPECTrecon.jl).

# ### Setup

# Packages needed here.

using SPECTrecon
using MIRTjim: jim, prompt
using Plots: scatter, scatter!, plot!, default
default(markerstrokecolor=:auto, markersize=3)

# The following line is helpful when running this example.jl file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

# ### Overview

# After rotating the image and the attenuation map,
# second step in SPECT image forward projection
# is to apply depth-dependent point spread function (PSF).
# Each (rotated) image plane
# is a certain distance from the SPECT detector
# and must be convolved with the 2D PSF appropriate
# for that plane.

# Because SPECT has relatively poor spatial resolution,
# the PSF is usually fairly wide,
# so convolution using FFT operations
# is typically more efficient
# than direct spatial convolution.

# Following other libraries like
# [FFTW.jl](https://github.com/JuliaMath/FFTW.jl),
# the PSF operations herein start with a `plan`
# where work arrays are preallocated
# for subsequent use.
# The `plan` is a `Vector` of `PlanPSF` objects:
# one for each thread.
# (Parallelism is across planes for a 3D image volume.)
# The number of threads defaults to `Threads.nthreads()`,
# but one can select any number
# and selecting more threads than number of cores
# empirically can reduce computation time.

# ### Example

# Start with a 3D image volume.

T = Float32 # work with single precision to save memory
nx = 64
nz = 30
image = zeros(T, nx, nx, nz) # ny = nx required
image[nx÷2, nx÷2, nz÷2] = 1
jim(image, "Original image")


# Create a synthetic depth-dependent PSF
nx_psf = 17
nview = 1 # for simplicity in this illustration
psf = zeros(nx_psf, nx_psf, nx, nview)

psf[:] .= 1 # todo: refine


# Now plan the PSF modeling
# by specifying
# * the image size (must be square)
# * the PSF size: must be `nx_psf × nx_psf × nx × nview`
# * the `DataType` used for the work arrays
# * the (maximum) number of threads.

plan = plan_rotate(nx, nz, nx_psf; T, nthread = Threads.nthreads())

# Here are the internals for the plan for the first thread:

plan[1]


# With this `plan` preallocated, now we can apply the depth-dependent PSF
# to the image volume (assumed already rotated here).

result = similar(image) # allocate memory for the result
# todo:
# apply_psf!(result, image, plan) # mutates the first argument
# jim(result, "After applying PSF")


# ### Adjoint

# To ensure adjoint consistency between SPECT forward- and back-projection,
# there is also an adjoint routine:

adj = similar(result)
# apply_psf_adj!(adj, result, plan
# jim(adj, "Adjoint of PSF modeling")


# The adjoint is *not* the same as the inverse
# so one does not expect the output here to match the original image!


# ### LinearMap

# One can form a linear map corresponding to PSF modeling using `LinearMapAA`.
# Perhaps the main purpose is simply for verifying adjoint correctness.

using LinearMapsAA: LinearMapAA

nx, nz, nx_psf = 20, 10, 5 # small size for illustration
plan = plan_rotate(nx, nz, nx_psf; T, nthread = Threads.nthreads())
idim = (nx,nx,nz)
odim = (nx,nx,nz)
#forw! = (y,x) -> apply_psf!(y, x, plan)
#back! = (x,y) -> apply_psf_adj!(x, y, plan)
#A = LinearMapAA(forw!, back!, (prod(odim),prod(idim)); T, odim, idim)

#Afull = Matrix(A)
#Aadj = Matrix(A')
#jim(cat(dims=3, Afull, Aadj'), "Linear map for PSF modeling and its adjoint")


# The following verifies adjoint consistency:
#@assert Afull ≈ Aadj'
