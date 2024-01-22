#=
# [SPECTrecon rotation](@id 2-rotate)

This page explains the image rotation portion of the Julia package
[`SPECTrecon.jl`](https://github.com/JuliaImageRecon/SPECTrecon.jl).
=#

#srcURL

# ## Setup

# Packages needed here.

using SPECTrecon: plan_rotate, imrotate!, imrotate_adj!
using MIRTjim: jim, prompt
using Plots: scatter, scatter!, plot!, default
default(markerstrokecolor=:auto, markersize=3)

# The following line is helpful when running this example.jl file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

#=
## Overview

The first step in SPECT image forward projection
is to rotate each slice of a 3D image volume
to the appropriate view angle.
In principle
one could use any of numerous candidate interpolation methods
for this task.
However, because emission images are nonnegative
and maximum likelihood methods
for SPECT image reconstruction
exploit that nonnegativity,
it is desirable to use interpolators
that preserve nonnegativity.
This constraint rules out quadratic and higher B-splines,
including the otherwise attractive cubic B-spline methods.
On the other hand, nearest-neighbor interpolation
(equivalent to 0th-order B-splines)
does not provide adequate image quality.
This leaves 1st-order interpolation methods
as the most viable options.

This package supports two 1st-order linear interpolators:
* 2D bilinear interpolation,
* a 3-pass rotation method based on 1D linear interpolation.

Because image rotation is done repeatedly
(for every slice of both the emission image and the attenuation map,
for both projection and back-projection,
and for multiple iterations),
it is important for efficiency
to use mutating methods
rather than to repeatedly make heap allocations.

Following other libraries like
[FFTW.jl](https://github.com/JuliaMath/FFTW.jl),
the rotation operations herein start with a `plan`
where work arrays are preallocated
for subsequent use.
The `plan` is a `Vector` of `PlanRotate` objects:
one for each thread.
(Parallelism is across slices for a 3D image volume.)
The number of threads defaults to `Threads.nthreads()`.


## Example

Start with a 3D image volume (just 2 slices here for simplicity).
=#

T = Float32 # work with single precision to save memory
image = zeros(T, 64, 64, 2)
image[30:50,20:30,1] .= 1
image[25:28,20:40,2] .= 1
jim(image, "Original image")

# Now plan the rotation
# by specifying
# * the image size `nx` (it must be square, so `ny=nx` implicitly)
# * the `DataType` used for the work arrays.

plan2 = plan_rotate(size(image, 1); T)

# Here are the internals for the plan for the first thread:

plan2[1]

# With this `plan` preallocated, now we can rotate the image volume,
# specifying the rotation angle in radians:

result2 = similar(image) # allocate memory for the result
imrotate!(result2, image, π/6, plan2) # mutates the first argument
jim(result2, "Rotated image by π/6 (2D bilinear)")

# The default, shown above, uses 2D bilinear interpolation for rotation.
# That default is the recommended approach because it is faster.

# Here is the 3-pass 1D interpolation approach,
# included mainly for checking consistency
# with the historical ASPIRE approach used in Matlab version of MIRT.

plan1 = plan_rotate(size(image, 1); T, method=:one)

# Here are the plan internals for the first thread:

plan1[1]

# The results of rotation using 3-pass 1D interpolation look quite similar:

result1 = similar(image)
imrotate!(result1, image, π/6, plan1)
jim(result1, "Rotated image by π/6 (3-pass 1D)")

# Here are the difference images for comparison.

jim(result1 - result2, "Difference images")


#=
## Adjoint

To ensure adjoint consistency between SPECT forward- and back-projection,
there is also an adjoint routine:
=#

adj2 = similar(result2)
imrotate_adj!(adj2, result2, π/6, plan2)
jim(adj2, "Adjoint image rotation (2D)")

adj1 = similar(result1)
imrotate_adj!(adj1, result1, π/6, plan1)
jim(adj1, "Adjoint image rotation (3-pass 1D)")



# The adjoint is *not* the same as the inverse
# so one does not expect the output here to match the original image!


#=
## LinearMap

One can form a linear map corresponding to image rotation using `LinearMapAA`.
An operator like this may be useful
as part of a motion-compensated image reconstruction method.
=#

using LinearMapsAA: LinearMapAA

nx = 20 # small size for illustration
r1 = plan_rotate(nx; T, nthread = 1, method=:two)[1]
r2 = plan_rotate(nx; T, nthread = 1, method=:one)[1]
idim = (nx,nx)
odim = (nx,nx)
forw! = (y,x) -> imrotate!(y, x, π/6, r1)
back! = (x,y) -> imrotate_adj!(x, y, π/6, r1)
A1 = LinearMapAA(forw!, back!, (prod(odim),prod(idim)); T, odim, idim)
forw! = (y,x) -> imrotate!(y, x, π/6, r2)
back! = (x,y) -> imrotate_adj!(x, y, π/6, r2)
A2 = LinearMapAA(forw!, back!, (prod(odim),prod(idim)); T, odim, idim)

Afull1 = Matrix(A1)
Aadj1 = Matrix(A1')
Afull2 = Matrix(A2)
Aadj2 = Matrix(A2')
jim(cat(dims=3, Afull1', Aadj1', Afull2', Aadj2'), "Linear map for 2D rotation and its adjoint")


# The following verify adjoint consistency:
@assert Afull1' ≈ Aadj1
@assert Afull2' ≈ Aadj2


# Applying this linear map to a 2D or 3D image performs rotation:

image2 = zeros(nx,nx); image2[4:6, 5:13] .= 1
jim(cat(dims=3, image2, A2 * image2), "Rotation via linear map: 2D")

# Here is 3D too.
# The `A2 * image3` here uses the advanced "operator" feature of
# [LinearMapsAA.jl](https://github.com/JeffFessler/LinearMapsAA.jl).

image3 = cat(dims=3, image2, image2')
jim(cat(dims=4, image3, A2 * image3), "Rotation via linear map: 3D")


# Examine row and column sums of linear map

scatter(xlabel="pixel index", ylabel="row or col sum")
scatter!(vec(sum(Afull1, dims=1)), label="dim1 sum1", marker=:x)
scatter!(vec(sum(Afull1, dims=2)), label="dim2 sum1", marker=:square)
scatter!(vec(sum(Afull2, dims=1)), label="dim1 sum2")
scatter!(vec(sum(Afull2, dims=2)), label="dim2 sum2")
