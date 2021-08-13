# fft_convolve.jl
# up to 8x faster and more efficient than ImageFiltering.imfilter!
import OffsetArrays

fftshift!(dst::AbstractArray, src::AbstractArray) = circshift!(dst, src, div.([size(src)...],2))
ifftshift!(dst::AbstractArray, src::AbstractArray) = circshift!(dst, src, div.([size(src)...],-2))
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

"""
    imfilter3!(padimg, ker, img_compl, ker_compl, fft_plan, ifft_plan)
    apply FFT convolution between padimg and kernel, assuming the kernel is already centered
"""
function imfilter3!(padimg::AbstractArray{<:Float32, 2},
                   ker::AbstractArray{<:Float32, 2},
                   img_compl::AbstractArray{Complex{T}, 2},
                   ker_compl::AbstractArray{Complex{T}, 2},
                   fft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan},
                   ifft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}) where T <: Real

    img_compl .= padimg
    ker_compl .= pad_it!(ker, size(img_compl))
    mul!(img_compl, fft_plan, img_compl)
    mul!(ker_compl, fft_plan, ker_compl)
    img_compl .*= ker_compl
    mul!(img_compl, ifft_plan, img_compl)
    padimg .= real.(fftshift!(ker_compl, img_compl))
end
