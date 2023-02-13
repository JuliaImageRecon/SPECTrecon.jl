# fft_convolve.jl

export fft_conv!, fft_conv_adj!
export fft_conv, fft_conv_adj

"""
    imfilterz!(plan)
FFT-based convolution of `plan.img_compl`
and kernel `plan.ker_compl` (not centered),
storing result in `plan.workmat`.
"""
function imfilterz!(plan::PlanPSF)
    mul!(plan.img_compl, plan.fft_plan, plan.img_compl)
    mul!(plan.ker_compl, plan.fft_plan, plan.ker_compl)
    broadcast!(*, plan.img_compl, plan.img_compl, plan.ker_compl)
    mul!(plan.img_compl, plan.ifft_plan, plan.img_compl)
    fftshift2!(plan.ker_compl, plan.img_compl)
    plan.workmat .= real.(plan.ker_compl)
    return plan.workmat
end


"""
    fft_conv!(output, img, ker, plan)
Convolve 2D image `img` with 2D (symmetric!) kernel `ker` using FFT,
storing the result in `output`.
"""
function fft_conv!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    ker::AbstractMatrix{<:RealU},
    plan::PlanPSF,
)
    @boundscheck size(output) == size(img) || throw("size output")
    @boundscheck size(img) == (plan.nx, plan.nz) || throw("size img")
    @boundscheck size(ker) == (plan.px, plan.pz) ||
        throw("size ker $(size(ker)) $(plan.px) $(plan.pz)")

    # filter image with a kernel, using replicate padding and fft convolution
    padrepl!(plan.img_compl, img, plan.padsize)

    pad2sizezero!(plan.ker_compl, ker, size(plan.ker_compl)) # zero pad kernel

    imfilterz!(plan)

    (M, N) = size(img)
    copyto!(output, (@view plan.workmat[plan.padsize[1] .+ (1:M),
                                        plan.padsize[3] .+ (1:N)]))
    return output
end


"""
    fft_conv(img, ker; T)
Convolve 2D image `img` with 2D (symmetric!) kernel `ker` using FFT.
"""
function fft_conv(
    img::AbstractMatrix{I},
    ker::AbstractMatrix{K};
    T::DataType = promote_type(I, K, Float32),
) where {I <: Number, K <: Number}

    ker ≈ reverse(ker, dims=:) || throw("asymmetric kernel")
    nx, nz = size(img)
    px, pz = size(ker)
    plan = plan_psf( ; nx, nz, px, pz, T, nthread = 1)[1]
    output = similar(Matrix{T}, size(img))
    fft_conv!(output, img, ker, plan)
    return output
end


"""
    fft_conv_adj!(output, img, ker, plan)
Adjoint of `fft_conv!`.
"""
function fft_conv_adj!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    ker::AbstractMatrix{<:RealU},
    plan::PlanPSF{T},
) where {T}

    @boundscheck size(output) == size(img) || throw("size output")
    @boundscheck size(img) == (plan.nx, plan.nz) || throw("size img")
    @boundscheck size(ker) == (plan.px, plan.pz) ||
        throw("size ker $(size(ker)) $(plan.px) $(plan.pz)")

    padzero!(plan.img_compl, img, plan.padsize) # pad the image with zeros
    pad2sizezero!(plan.ker_compl, ker, size(plan.ker_compl)) # pad the kernel with zeros

    imfilterz!(plan)
    (M, N) = size(img)
    # adjoint of replicate padding
    plan.workvecz .= zero(T)
    for i in 1:plan.padsize[1]
        plus2di!(plan.workvecz, plan.workmat, i)
    end
    plus1di!(plan.workmat, plan.workvecz, 1+plan.padsize[1])

    plan.workvecz .= zero(T)
    for i in (plan.padsize[1]+M+1):size(plan.workmat, 1)
        plus2di!(plan.workvecz, plan.workmat, i)
    end
    plus1di!(plan.workmat, plan.workvecz, M+plan.padsize[1])

    plan.workvecx .= zero(T)
    for j in 1:plan.padsize[3]
        plus2dj!(plan.workvecx, plan.workmat, j)
    end
    plus1dj!(plan.workmat, plan.workvecx, 1+plan.padsize[3])

    plan.workvecx .= zero(T)
    for j in (plan.padsize[3]+N+1):size(plan.workmat, 2)
        plus2dj!(plan.workvecx, plan.workmat, j)
    end
    plus1dj!(plan.workmat, plan.workvecx, N+plan.padsize[3])

    copyto!(output,
        (@view plan.workmat[(plan.padsize[1]+1):(plan.padsize[1]+M),
                            (plan.padsize[3]+1):(plan.padsize[3]+N)]),
    )

    return output
end


"""
    fft_conv_adj(img, ker; T)
Adjoint of `fft_conv`.
"""
function fft_conv_adj(
    img::AbstractMatrix{I},
    ker::AbstractMatrix{K};
    T::DataType = promote_type(I, K, Float32),
) where {I <: Number, K <: Number}

    ker ≈ reverse(ker, dims=:) || throw("asymmetric kernel")
    nx, nz = size(img)
    px, pz = size(ker)
    plan = plan_psf( ; nx, nz, px, pz, T, nthread = 1)[1]
    output = similar(Matrix{T}, size(img))
    fft_conv_adj!(output, img, ker, plan)
    return output
end


"""
    fft_conv!(output, image3, ker3, plans)
Mutating version of convolving a 3D `image3`
with a set of 2D symmetric kernels
stored in 3D array `ker3`
using `foreach`.
"""
function fft_conv!(
    output::AbstractArray{<:RealU,3},
    image3::AbstractArray{<:RealU,3},
    ker3::AbstractArray{<:RealU,3},
    plans::Vector{<:PlanPSF},
)

    size(output) == size(image3) || throw(DimensionMismatch())

    fun = y -> fft_conv!(
            (@view output[:, y, :]),
            (@view image3[:, y, :]),
            (@view ker3[:, :, y]),
            plans[Threads.threadid()],
        )

    ntasks = length(plans)
    Threads.foreach(fun, foreach_setup(1:size(image3, 2)); ntasks)

    return output
end


"""
    fft_conv_adj!(output, image3, ker3, plans)
Adjoint of `fft_conv`.
"""
function fft_conv_adj!(
    output::AbstractArray{<:RealU,3},
    image3::AbstractArray{<:RealU,3},
    ker3::AbstractArray{<:RealU,3},
    plans::Vector{<:PlanPSF},
)

    size(output) == size(image3) || throw(DimensionMismatch())

    fun = y -> fft_conv_adj!(
            (@view output[:, y, :]),
            (@view image3[:, y, :]),
            (@view ker3[:, :, y]),
            plans[Threads.threadid()],
        )

    ntasks = length(plans)
    Threads.foreach(fun, foreach_setup(1:size(image3, 2)); ntasks)

    return output
end


"""
    fft_conv_adj2!(output, image2, ker3, plans)
Mutating version of adjoint of convolving a 2D `image2`
with each 2D kernel in the 3D array `ker3`.
"""
function fft_conv_adj2!(
    output::AbstractArray{<:RealU,3},
    image2::AbstractMatrix{<:RealU},
    ker3::AbstractArray{<:RealU,3},
    plans::Vector{<:PlanPSF},
)

    size(output, 1) == size(image2, 1) || throw("size 1")
    size(output, 3) == size(image2, 2) || throw("size 2")

    fun = y -> fft_conv_adj!(
            (@view output[:, y, :]),
            image2,
            (@view ker3[:, :, y]),
            plans[Threads.threadid()],
        )

    ntasks = length(plans)
    Threads.foreach(fun, foreach_setup(1:size(output, 2)); ntasks)

    return output
end
