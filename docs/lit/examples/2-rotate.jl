#---------------------------------------------------------
# # [SPECTrecon rotation](@id 2-rotate)
#---------------------------------------------------------

# This page explains the image rotation portion of the Julia package
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

# The first step in SPECT image forward projection
# is to rotate each slice of a 3D image volume
# to the appropriate view angle.
# In principle
# one could use any of numerous candidate interpolation methods
# for this task.
# However, because emission images are nonnegative
# and maximum likelihood methods
# for SPECT image reconstruction
# exploit that nonnegativity,
# it is desirable to use interpolators
# that preserve nonnegativity.
# This constraint rules out quadratic and higher B-splines,
# including the otherwise attractive cubic B-spline methods.
# On the other hand, nearest-neighbor interpolation
# (equivalent to 0th-order B-splines)
# does not provide adequate image quality.
# This leaves 1st-order interpolation methods
# as the most viable options.

# This package supports two 1st-order linear interpolators:
# * 2D bilinear interpolation
# * a 3-pass rotation method based on 1D linear interpolation.

# Because image rotation is done repeatedly
# (for every slice of both the emission image and the attenuation map,
# for both projection and back-projection,
# and for multiple iterations)
# it is important for efficiency
# to use mutating functions
# rather than to repeatedly make heap allocations.

# Following other libraries like
# [FFTW.jl](https://github.com/JuliaMath/FFTW.jl),
# the rotation operations herein start with a `plan`
# where work arrays are preallocated
# for subsequent use.

# This

# ### Example

# Start with a 3D image volume (just 2 slices here for simplicity)

T = Float32 # work with single precision to save memory
image = zeros(T, 64, 64, 2)
image[30:50,20:30,1] .= 1
image[25:28,20:40,2] .= 1
jim(image, "Original image")

# Now plan the rotation

rplan = plan_rotate(size(image, 1); T, nthread = Threads.nthreads())

# Here are the plan internals:

rplan[1]

# With this `plan` preallocated, now we can rotate the image volume: 

result = similar(image)
imrotate!(result, image, π/6, rplan)
jim(result, "Rotated image by π/6 (2D bilinear)")


# The rotation angle can (and should!) be a value with units.
# (todo: but it is failing currently)

# using UnitfulRecipes
# using Unitful: °


# imrotate!(result, image, 10°, rplan)
# jim(result, "Rotated image by 10°")

# The default, shown above, is 2D bilinear iterpolation for rotation.
# That is the recommended approach because it is faster.

# Here is the 3-pass 1D interpolation approach,
# included mainly for checking consistency with the ASPIRE approach.

plan1 = plan_rotate(size(image, 1); T, nthread = Threads.nthreads(), method=:one)

# Here are the plan internals:

plan1[1]

# And the results look quite similar:

result1 = similar(image)
imrotate!(result1, image, π/6, rplan)
jim(result1, "Rotated image by π/6 (3-pass 1D)")


### Adjoint

# To ensure adjoint consistency between SPECT forward- and back-projection,
# there is also an adjoint routine:

imagea = similar(result)
imrotate_adj!(imagea, result, π/6, rplan)
jim(imagea, "Adjoint image rotation")


# The adjoint is *not* the same as the inverse
# so one does not expect the output here to be the original image!


### LinearMap

# One can form a linear map corresponding to image rotation using `LinearMapAA`.
# An operator like this may be useful
# as part of a motion-compensated image reconstruction method.

using LinearMapsAA: LinearMapAA

nx = 20 # small size for illustration
plan0 = plan_rotate(nx; T, nthread = 1, method=:two)[1]
idim = (nx,nx)
odim = (nx,nx)
forw! = (y,x) -> imrotate!(y, x, π/6, plan0)
back! = (x,y) -> imrotate_adj!(x, y, π/6, plan0)
A = LinearMapAA(forw!, back!, (prod(odim),prod(idim)); T, odim, idim)

Afull = Matrix(A)
Aadj = Matrix(A')
jim(cat(dims=3,Afull,Aadj), "Linear map for 2D rotation and its adjoint")


# The following line verifies adjoint consistency:
@assert isapprox(Afull', Aadj)
