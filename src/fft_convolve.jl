# fft_convolve.jl
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
    pad2sizezero!(ker_compl, ker, size(img_compl))
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



"""
    imfilter3_adj!(output, img_compl, kerev, ker_compl, fft_plan, ifft_plan)
    apply FFT convolution between padimg and *REVERSED* kernel,
    assuming the kernel is already centered
    and is already be in reversed order.
"""
function imfilter3_adj!(output::AbstractArray{<:Real, 2},
                        img_compl::AbstractArray{Complex{T}, 2},
                        kerev::AbstractArray{<:Real, 2}, # input kernel should already be in reversed order
                        ker_compl::AbstractArray{Complex{T}, 2},
                        fft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan},
                        ifft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}) where T <: Real

    # ker_compl .= pad_it!(ker, size(img_compl))
    pad2sizezero!(ker_compl, kerev, size(img_compl))
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
# @btime imfilter3_adj!(output, img_compl, ker, ker_compl, fft_plan, ifft_plan)
# 32.519 μs (0 allocations: 0 bytes)
# y = imfilter(img, centered(ker))
# plot(jim(output), jim(y), jim(output - y))



"""
    fft_conv!(output, workmat, img, ker, fftpadsize, img_compl, ker_compl, fft_plan, ifft_plan)
"""
function fft_conv!(output::AbstractArray{<:Real, 2},
                  workmat::AbstractArray{<:Real, 2},
                  img::AbstractArray{<:Real, 2},
                  ker::AbstractArray{<:Real, 2},
                  fftpadsize::NTuple{4, <:Int},
                  img_compl::AbstractArray{Complex{T}, 2},
                  ker_compl::AbstractArray{Complex{T}, 2},
                  fft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan},
                  ifft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}) where T <: Real
    # filter the image with a kernel, using replicate padding and fft convolution
    padrepl!(img_compl, img, fftpadsize)
    imfilter3!(workmat, img_compl, ker, ker_compl, fft_plan, ifft_plan)
    (M, N) = size(img)
    copyto!(output, (@view workmat[fftpadsize[1]+1:fftpadsize[1]+M,
                                   fftpadsize[3]+1:fftpadsize[3]+N]))
end

# Test code:
# M = 200
# N = 64
# T = Float32
# img = zeros(T, M, N)
# img[20:150, 20:40] .= rand(131,21)
# output = similar(img)
# ker = ones(T, 3, 3) / 9
# fftpadsize = (28, 28, 32, 32)
#
# img_compl = zeros(Complex{T}, 256, 128)
# workmat = zeros(T, 256, 128)
# ker_compl = similar(img_compl)
# fft_plan = plan_fft!(img_compl)
# ifft_plan = plan_ifft!(img_compl)
# fft_conv!(output, workmat, img, ker, fftpadsize,
#           img_compl, ker_compl, fft_plan, ifft_plan)
# @btime fft_conv!(output, workmat, img, ker, fftpadsize,
#           img_compl, ker_compl, fft_plan, ifft_plan)
# 585.016 μs (0 allocations: 0 bytes)

"""
    fft_conv_adj!(output, workmat, workvec1, workvec2, img, ker,
                  fftpadsize, img_compl, ker_compl, fft_plan, ifft_plan)
"""
function fft_conv_adj!(output::AbstractArray{<:Real, 2},
                       workmat::AbstractArray{<:Real, 2},
                       workvec1::AbstractVector{<:Real},
                       workvec2::AbstractVector{<:Real},
                       img::AbstractArray{<:Real, 2},
                       ker::AbstractArray{<:Real, 2},
                       fftpadsize::NTuple{4, <:Int},
                       img_compl::AbstractArray{Complex{T}, 2},
                       ker_compl::AbstractArray{Complex{T}, 2},
                       fft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan},
                       ifft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}) where T <: Real

    padzero!(img_compl, img, fftpadsize)
    imfilter3_adj!(workmat, img_compl, ker, ker_compl, fft_plan, ifft_plan)
    (M, N) = size(img)
    # adjoint of replicate padding
    workvec1 .= 0
    for i = 1:fftpadsize[1]
        plus2di!(workvec1, workmat, i)
    end
    plus1di!(workmat, workvec1, 1+fftpadsize[1])

    workvec1 .= 0
    for i = fftpadsize[1]+M+1:size(workmat, 1)
        plus2di!(workvec1, workmat, i)
    end
    plus1di!(workmat, workvec1, M+fftpadsize[1])
    workvec2 .= 0
    for j = 1:fftpadsize[3]
        plus2dj!(workvec2, workmat, j)
    end
    plus1dj!(workmat, workvec2, 1+fftpadsize[3])

    workvec2 .= 0
    for j = fftpadsize[3]+N+1:size(workmat, 2)
        plus2dj!(workvec2, workmat, j)
    end
    plus1dj!(workmat, workvec2, N+fftpadsize[3])

    copyto!(output, (@view workmat[fftpadsize[1]+1:fftpadsize[1]+M,
                                   fftpadsize[3]+1:fftpadsize[3]+N]))
end

# Test code
# M = 200
# N = 64
# T = Float32
# img = zeros(T, M, N)
# img[20:150, 20:40] .= rand(131,21)
# output = similar(img)
# ker = ones(T, 3, 3) / 9
# fftpadsize = (28, 28, 32, 32)
#
# img_compl = zeros(Complex{T}, 256, 128)
# workmat = zeros(T, 256, 128)
# workvec1 = zeros(T, 128)
# workvec2 = zeros(T, 256)
# ker_compl = similar(img_compl)
# fft_plan = plan_fft!(img_compl)
# ifft_plan = plan_ifft!(img_compl)
# fft_conv_adj!(output, workmat, workvec1, workvec2, img, ker, fftpadsize,
#           img_compl, ker_compl, fft_plan, ifft_plan)
# @btime fft_conv_adj!(output, workmat, workvec1, workvec2, img, ker, fftpadsize,
#           img_compl, ker_compl, fft_plan, ifft_plan)
# 594.788 μs (0 allocations: 0 bytes)
