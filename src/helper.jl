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
Base.@propagate_inbounds function padzero!(output::AbstractMatrix{<:Real},
                                           img::AbstractMatrix{<:Real},
                                           pad_x::Int,
                                           pad_y::Int)

        @assert size(output) == size(img) .+ (2*pad_x, 2*pad_y)
        M, N = size(img)
        output .= 0
        for j = pad_y + 1:pad_y + N, i = pad_x + 1:pad_x + M
                @inbounds output[i, j] = img[i - pad_x, j - pad_y]
        end
        # (@view output[pad_x + 1:pad_x + M, pad_y + 1:pad_y + N]) .= img
end

# Test code:
# x = randn(7,5)
# y = randn(3,3)
# @btime padzero!(x, y, 2, 1)
# 28.385 ns (0 allocations: 0 bytes)

Base.@propagate_inbounds function pad2size!(output::AbstractMatrix{<:Complex},
                                            img::AbstractMatrix{<:Complex},
                                            padsize::Tuple{<:Int})
    @assert size(output) == padsize
    dims = size(img)
    pad_dims = ceil.(Int, (padsize .- dims) ./ 2)
    output .= 0
    for j = pad_dims[2]+1:pad_dims[2]+dims[2], i = pad_dims[1]+1:pad_dims[1]+dims[1]
        @inbounds output[i, j] = img[i - pad_dims[1], j - pad_dims[2]]
    end
    # (@view output[pad_dims[1]+1:pad_dims[1]+dims[1],
    #               pad_dims[2]+1:pad_dims[2]+dims[2]]) .= img
end

function pad_it!(X::AbstractArray,
                padsize::Tuple)

    dims = size(X)
    return OffsetArrays.no_offset_view(
        BorderArray(X,
            Fill(0,
               (ceil.(Int, (padsize .- dims) ./ 2)),
               (floor.(Int, (padsize .- dims) ./ 2)),
            )
        )
    )
end

# Test code:
# ker = randn(5,5)
# padsize = (64, 64)
# z = randn(padsize)
# pad2size!(z, ker, padsize)
# isequal(pad_it!(ker, padsize), z)
# @btime pad2size!(z, ker, padsize)
# 451.000 ns (0 allocations: 0 bytes)

fftshift!(dst::AbstractArray{<:Real}, src::AbstractArray{<:Real}) = circshift!(dst, src, div.([size(src)...],2))
# Test code:
# b = [1 2;3 4]
# x = [b 2*b;3*b 4*b]
# y = similar(x)
# @btime fftshift!(y, x)

ifftshift!(dst::AbstractArray, src::AbstractArray) = circshift!(dst, src, div.([size(src)...],-2))
