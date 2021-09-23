# fft_convolve.jl
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


"""
    imfilter3!(output, img_compl, ker, ker_compl, fft_plan, ifft_plan)
    apply FFT convolution between padimg and kernel, assuming the kernel is already centered
"""
function imfilter3!(output::AbstractArray{<:Real, 2},
                   img_compl::AbstractArray{Complex{T}, 2},
                   ker::AbstractArray{<:Real, 2},
                   ker_compl::AbstractArray{Complex{T}, 2},
                   fft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan},
                   ifft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}) where T <: Real

    # ker_compl .= pad_it!(ker, size(img_compl))
    pad2size!(ker_compl, ker, size(img_compl))
    mul!(img_compl, fft_plan, img_compl)
    mul!(ker_compl, fft_plan, ker_compl)
    img_compl .*= ker_compl
    mul!(img_compl, ifft_plan, img_compl)
    recenter2d!(ker_compl, img_compl)
    output .= real.(ker_compl)
end

# Test code:
# N = 64
# T = Float32
# img = zeros(T, N, N)
# img[20:50, 20:40] .= rand(31,21)
# output = similar(img)
# ker = ones(T, 3, 3) / 9
# img_compl = similar(img, Complex{T})
# ker_compl = similar(img_compl)
# fft_plan = plan_fft!(img_compl)
# ifft_plan = plan_ifft!(img_compl)
# copyto!(img_compl, img)
# @btime imfilter3!(output, img_compl, ker, ker_compl, fft_plan, ifft_plan)
# 32.519 μs (0 allocations: 0 bytes)
# y = imfilter(img, centered(ker))
# plot(jim(output), jim(y), jim(output - y))
