# helper.jl

# A lot of helper functions
using LinearAlgebra
using LazyAlgebra, TwoDimensional
using LinearInterpolators
using InterpolationKernels
using OffsetArrays
using ImageFiltering
using FFTW

Power2 = x -> 2^(ceil(Int, log2(x)))
_padleft(mumap, psfs) = ceil(Int, (Power2(size(mumap, 1) + size(psfs, 1) - 1) - size(mumap, 1)) / 2)
_padright(mumap, psfs) = floor(Int, (Power2(size(mumap, 1) + size(psfs, 1) - 1) - size(mumap, 1)) / 2)
_padup(mumap, psfs) = ceil(Int, (Power2(size(mumap, 3) + size(psfs, 2) - 1) - size(mumap, 3)) / 2)
_paddown(mumap, psfs) = floor(Int, (Power2(size(mumap, 3) + size(psfs, 2) - 1) - size(mumap, 3)) / 2)
