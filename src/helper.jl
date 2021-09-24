# helper.jl

# A lot of helper functions
using LinearAlgebra
using LazyAlgebra, TwoDimensional
using LinearInterpolators
using InterpolationKernels
using OffsetArrays
using ImageFiltering
using FFTW

const RealU = Number # Union{Real, Unitful.Length}
Power2 = x -> 2^(ceil(Int, log2(x)))
_padup(mumap, psfs) = ceil(Int, (Power2(size(mumap, 1) + size(psfs, 1) - 1) - size(mumap, 1)) / 2)
_paddown(mumap, psfs) = floor(Int, (Power2(size(mumap, 1) + size(psfs, 1) - 1) - size(mumap, 1)) / 2)
_padleft(mumap, psfs) = ceil(Int, (Power2(size(mumap, 3) + size(psfs, 2) - 1) - size(mumap, 3)) / 2)
_padright(mumap, psfs) = floor(Int, (Power2(size(mumap, 3) + size(psfs, 2) - 1) - size(mumap, 3)) / 2)

"""
    padzero!(output, img, pad_x, pad_y)
inplace version of padding a 2D image by filling zeros
output has size (size(img, 1) + padsize[1] + padsize[2], size(img, 2) + padsize[3] + padsize[4])
"""
Base.@propagate_inbounds function padzero!(output::AbstractMatrix{<:Any},
                                           img::AbstractMatrix{<:Any},
                                           padsize::NTuple{4, <:Int}, # up, down, left, right
                                           )

        @assert size(output) == size(img) .+ (padsize[1] + padsize[2], padsize[3] + padsize[4])
        M, N = size(img)
        output .= 0
        for j = padsize[3] + 1:padsize[3] + N, i = padsize[1] + 1:padsize[1] + M
                @inbounds output[i, j] = img[i - padsize[1], j - padsize[3]]
        end
        # (@view output[pad_x + 1:pad_x + M, pad_y + 1:pad_y + N]) .= img
end

# Test code:
# x = randn(7,5)
# y = randn(3,3)
# padzero!(x, y, (2, 2, 1, 1))
# z = OffsetArrays.no_offset_view(BorderArray(y, Fill(0, (2, 1), (2, 1))))
# isequal(x, z)
# @btime padzero!(x, y, (2, 2, 1, 1))
# 29.030 ns (0 allocations: 0 bytes)

Base.@propagate_inbounds function padrepl!(output::AbstractMatrix{<:Any},
                                           img::AbstractMatrix{<:Any},
										   padsize::NTuple{4, <:Int}, # up, down, left, right
										   )

        @assert size(output) == size(img) .+ (padsize[1] + padsize[2], padsize[3] + padsize[4])
        M, N = size(img)

		for j = 1:size(output, 2), i = 1:size(output, 1)
			@inbounds output[i, j] = img[clamp(i - padsize[1], 1, M), clamp(j - padsize[3], 1, N)]
		end

end

# Test code:
# x = randn(10,9)
# y = randn(5,4)
# padrepl!(x, y, (1, 4, 3, 2))
# # up, down, left, right
# z = OffsetArrays.no_offset_view(BorderArray(y, Pad(:replicate, (1, 3), (4, 2))))
# # up, left, down, right
# isequal(x, z)
# @btime padrepl!(x, y, (1, 4, 3, 2))
# 83.394 ns (0 allocations: 0 bytes)

Base.@propagate_inbounds function pad2sizezero!(output::AbstractMatrix{<:Any},
                                            	img::AbstractMatrix{<:Any},
                                            	padsize::Tuple{<:Int, <:Int})
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
# pad2sizezero!(z, ker, padsize)
# isequal(pad_it!(ker, padsize), z)
# @btime pad2sizezero!(z, ker, padsize)
# 451.000 ns (0 allocations: 0 bytes)

fftshift!(dst::AbstractArray, src::AbstractArray) = circshift!(dst, src, div.([size(src)...],2))
# Test code:
# b = [1 2;3 4]
# x = [b 2*b;3*b 4*b]
# y = similar(x)
# @btime fftshift!(y, x)

ifftshift!(dst::AbstractArray, src::AbstractArray) = circshift!(dst, src, div.([size(src)...],-2))

"""
    recenter2d!(dst, src)
    the same as fftshift in 2d, but zero allocation
"""
function recenter2d!(dst::AbstractMatrix{<:Any},
                     src::AbstractMatrix{<:Any})
        @assert iseven(size(src, 1)) && iseven(size(src, 2))
        m, n = div.(size(src), 2)
        for j = 1:n, i = 1:m
            dst[i, j] = src[m+i, n+j]
        end
        for j = n+1:2n, i = 1:m
            dst[i, j] = src[m+i, j-n]
        end
        for j = 1:n, i = m+1:2m
            dst[i, j] = src[i-m, j+n]
        end
        for j = n+1:2n, i = m+1:2m
            dst[i, j] = src[i-m, j-n]
        end
end

# Test code:
# x = randn(100, 80)
# y = similar(x)
# z = similar(x)
# fftshift!(y, x)
# recenter2d!(z, x)
# isequal(y, z)


Base.@propagate_inbounds function plus1di!(mat2d::AbstractArray{<:Any, 2},
										   mat1d::AbstractArray{<:Any, 1},
										   i::Int) # mat2d[i, :] += mat1d
	@boundscheck (size(mat1d, 1) == size(mat2d, 2) || throw("size2"))
	@boundscheck (1 ≤ i ≤ size(mat2d, 1) || throw("bad i"))
	for m in 1:size(mat2d, 2)
		@inbounds mat2d[i, m] += mat1d[m]
	end
end

# Test code:
# x = randn(4, 64)
# v = randn(64)
# y = x[2, :] .+ v
# plus1di!(x, v, 2)
# isequal(x[2, :], y)
# @btime plus1di!(x, v, 2)
# 45.022 ns (0 allocations: 0 bytes)


Base.@propagate_inbounds function plus1dj!(mat2d::AbstractArray{<:Any, 2},
										   mat1d::AbstractArray{<:Any, 1},
										   j::Int) # mat2d[:, j] += mat1d
	@boundscheck (size(mat1d, 1) == size(mat2d, 1) || throw("size1"))
	@boundscheck (1 ≤ j ≤ size(mat2d, 2) || throw("bad j"))
	for n in 1:size(mat2d, 1)
		@inbounds mat2d[n, j] += mat1d[n]
	end
end

# Test code:
# x = randn(64, 4)
# v = randn(64)
# y = x[:, 2] .+ v
# plus1dj!(x, v, 2)
# isequal(x[:, 2], y)
# @btime plus1dj!(x, v, 2)
# 23.168 ns (0 allocations: 0 bytes)



Base.@propagate_inbounds function plus2di!(mat1d::AbstractArray{<:Any, 1},
										   mat2d::AbstractArray{<:Any, 2},
										   i::Int) # mat1d += mat2d[i,:]
	@boundscheck (size(mat1d, 1) == size(mat2d, 2) || throw("size2"))
	@boundscheck (1 ≤ i ≤ size(mat2d, 1) || throw("bad i"))
	for m in 1:size(mat2d, 2)
		@inbounds mat1d[m] += mat2d[i, m]
	end
end

# Test code:
# x = randn(64)
# v = randn(4, 64)
# y = x .+ v[2, :]
# plus2di!(x, v, 2)
# isequal(x, y)
# @btime plus2di!(x, v, 2)
# 46.227 ns (0 allocations: 0 bytes)

Base.@propagate_inbounds function plus2dj!(mat1d::AbstractArray{<:Any, 1},
										   mat2d::AbstractArray{<:Any, 2},
										   j::Int) # mat1d += mat2d[:,j]
	@boundscheck (size(mat1d, 1) == size(mat2d, 1) || throw("size1"))
	@boundscheck (1 ≤ j ≤ size(mat2d, 2) || throw("bad j"))
	for n in 1:size(mat2d, 1)
		@inbounds mat1d[n] += mat2d[n, j]
	end
end

# Test code:
# x = randn(64)
# v = randn(64, 4)
# y = x .+ v[:, 2]
# plus2dj!(x, v, 2)
# isequal(x, y)
# @btime plus2dj!(x, v, 2)
# 25.177 ns (0 allocations: 0 bytes)

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
