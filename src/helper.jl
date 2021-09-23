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
Base.@propagate_inbounds function padzero!(output::AbstractMatrix{<:Any},
                                           img::AbstractMatrix{<:Any},
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

Base.@propagate_inbounds function pad2size!(output::AbstractMatrix{<:Any},
                                            img::AbstractMatrix{<:Any},
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

function pad_it!(X::AbstractArray{<:Any},
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

fftshift!(dst::AbstractArray, src::AbstractArray) = circshift!(dst, src, div.([size(src)...],2))
# Test code:
# b = [1 2;3 4]
# x = [b 2*b;3*b 4*b]
# y = similar(x)
# @btime fftshift!(y, x)

ifftshift!(dst::AbstractArray, src::AbstractArray) = circshift!(dst, src, div.([size(src)...],-2))


Base.@propagate_inbounds function plus3di!(mat2d::AbstractArray{<:Any, 2},
										   mat3d::AbstractArray{<:Any, 3},
										   i::Int) # mat2d += mat3d[i,:,:]
	@boundscheck (size(mat2d, 1) == size(mat3d, 2) || throw("size2"))
	@boundscheck (size(mat2d, 2) == size(mat3d, 3) || throw("size3"))
	@boundscheck (1 ≤ i ≤ size(mat3d, 1) || throw("bad i"))
	for n in 1:size(mat2d, 2), m in 1:size(mat2d, 1)
		@inbounds mat2d[m, n] += mat3d[i, m, n]
	end
end

# Test code:
# x = randn(64, 64)
# v = randn(4, 64, 64)
# y = x .+ v[2, :, :]
# plus3di!(x, v, 2)
# isequal(x, y)
# @btime plus3di!(x, v, 2)
# 2.368 μs (0 allocations: 0 bytes)


Base.@propagate_inbounds function plus3dj!(mat2d::AbstractArray{<:Any, 2},
										   mat3d::AbstractArray{<:Any, 3},
										   j::Int) # mat2d += mat3d[:,j,:]
	@boundscheck (size(mat2d, 1) == size(mat3d, 1) || throw("size1"))
	@boundscheck (size(mat2d, 2) == size(mat3d, 3) || throw("size3"))
	@boundscheck (1 ≤ j ≤ size(mat3d, 2) || throw("bad j"))
	for n in 1:size(mat2d, 2), m in 1:size(mat2d, 1)
		@inbounds mat2d[m, n] += mat3d[m, j, n]
	end
end

# Test code:
# x = randn(64, 64)
# v = randn(64, 4, 64)
# y = x .+ v[:, 2, :]
# plus3dj!(x, v, 2)
# isequal(x, y)
# @btime plus3dj!(x, v, 2)
# 585.718 ns (0 allocations: 0 bytes)


Base.@propagate_inbounds function plus3dk!(mat2d::AbstractArray{<:Any, 2},
										   mat3d::AbstractArray{<:Any, 3},
										   k::Int) # mat2d += mat3d[:,:,k]
	@boundscheck (size(mat2d, 1) == size(mat3d, 1) || throw("size1"))
	@boundscheck (size(mat2d, 2) == size(mat3d, 2) || throw("size2"))
	@boundscheck (1 ≤ k ≤ size(mat3d, 3) || throw("bad k"))
	for n in 1:size(mat2d, 2), m in 1:size(mat2d, 1)
		@inbounds mat2d[m, n] += mat3d[m, n, k]
	end
end

# Test code:
# x = randn(64, 64)
# v = randn(64, 64, 4)
# y = x .+ v[:, :, 2]
# plus3dk!(x, v, 2)
# isequal(x, y)
# @btime plus3dk!(x, v, 2)
# 595.462 ns (0 allocations: 0 bytes)
