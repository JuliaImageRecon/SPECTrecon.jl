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

"""
    padzero!(output, img, pad_x, pad_y)
inplace version of padding a 2D image by filling zeros
output has size (size(img, 1) + 2 * pad_x, size(img, 2) + 2 * pad_y)
"""
function padzero!(output::AbstractMatrix{<:Real},
                  img::AbstractMatrix{<:Real},
                  pad_x::Int,
                  pad_y::Int)
        @assert size(output) == size(img) .+ (2*pad_x, 2*pad_y)
        M, N = size(img)
        (@view output[pad_x + 1:pad_x + M, pad_y + 1:pad_y + N]) .= img
        (@view output[1:pad_x, :]) .= 0
        (@view output[pad_x + M + 1:end, :]) .= 0
        (@view output[:, 1:pad_y]) .= 0
        (@view output[:, pad_y + N + 1:end]) .= 0
end

# Test code:
# x = randn(7,5)
# y = randn(3,3)
# @btime padzero!(x, y, 2, 1)
# 80.762 ns (0 allocations: 0 bytes)
