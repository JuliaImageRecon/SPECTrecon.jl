#=
# [SPECTrecon PSF](@id 3-psf)

This page explains the PSF portion of the Julia package
[`SPECTrecon.jl`](https://github.com/JuliaImageRecon/SPECTrecon.jl).

This page was generated from a single Julia file:
[3-psf.jl](@__REPO_ROOT_URL__/3-psf.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](https://nbviewer.org/) here:
#md # [`3-psf.ipynb`](@__NBVIEWER_ROOT_URL__/3-psf.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`3-psf.ipynb`](@__BINDER_ROOT_URL__/3-psf.ipynb).

# ## Setup

# Packages needed here.

using SPECTrecon
using MIRTjim: jim, prompt
using Plots: scatter, scatter!, plot!, default
default(markerstrokecolor=:auto, markersize=3)

# The following line is helpful when running this example.jl file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

#=
## Overview

After rotating the image and the attenuation map,
second step in SPECT image forward projection
is to apply depth-dependent point spread function (PSF).
Each (rotated) image plane
is a certain distance from the SPECT detector
and must be convolved with the 2D PSF appropriate
for that plane.

Because SPECT has relatively poor spatial resolution,
the PSF is usually fairly wide,
so convolution using FFT operations
is typically more efficient
than direct spatial convolution.

Following other libraries like
[FFTW.jl](https://github.com/JuliaMath/FFTW.jl),
the PSF operations herein start with a `plan`
where work arrays are preallocated
for subsequent use.
The `plan` is a `Vector` of `PlanPSF` objects:
one for each thread.
(Parallelism is across planes for a 3D image volume.)
The number of threads defaults to `Threads.nthreads()`.


## Example

Start with a 3D image volume.
=#

T = Float32 # work with single precision to save memory
nx = 32
nz = 30
image = zeros(T, nx, nx, nz) # ny = nx required
image[1nx÷4, 1nx÷4, 3nz÷4] = 1
image[2nx÷4, 2nx÷4, 2nz÷4] = 1
image[3nx÷4, 3nx÷4, 1nz÷4] = 1
jim(image, "Original image")


# Create a synthetic gaussian depth-dependent PSF for a single view

px = 11
nview = 1 # for simplicity in this illustration
psf = repeat(psf_gauss( ; ny=nx, px), 1, 1, 1, nview)
jim(psf, "PSF for each of $nx planes")


#=
Now plan the PSF modeling
by specifying
* the image size (must be square)
* the PSF size: must be `px × pz × ny × nview`
* the `DataType` used for the work arrays.
=#

plan = plan_psf( ; nx, nz, px, T)

# Here are the internals for the plan for the first thread:

plan[1]


# With this `plan` pre-allocated, now we can apply the depth-dependent PSF
# to the image volume (assumed already rotated here).

result = similar(image) # allocate memory for the result
fft_conv!(result, image, psf[:,:,:,1], plan) # mutates the first argument
jim(result, "After applying PSF")


#=
## Adjoint

To ensure adjoint consistency between SPECT forward- and back-projection,
there is also an adjoint routine:
=#

adj = similar(result)
fft_conv_adj!(adj, result, psf[:,:,:,1], plan)
jim(adj, "Adjoint of PSF modeling")


# The adjoint is *not* the same as the inverse
# so one does not expect the output here to match the original image!


#=
## LinearMap

One can form a linear map corresponding to PSF modeling using `LinearMapAA`.
Perhaps the main purpose is simply for verifying adjoint correctness.
=#

using LinearMapsAA: LinearMapAA

nx, nz, px = 10, 7, 5 # small size for illustration
psf3 = psf_gauss( ; ny=nx, px)
plan = plan_psf( ; nx, nz, px, T)
idim = (nx,nx,nz)
odim = (nx,nx,nz)
forw! = (y,x) -> fft_conv!(y, x, psf3, plan)
back! = (x,y) -> fft_conv_adj!(x, y, psf3, plan)
A = LinearMapAA(forw!, back!, (prod(odim),prod(idim)); T, odim, idim)

Afull = Matrix(A)
Aadj = Matrix(A')
jim(cat(dims=3, Afull, Aadj'), "Linear map for PSF modeling and its adjoint")


# The following check verifies adjoint consistency:

@assert Afull ≈ Aadj'
