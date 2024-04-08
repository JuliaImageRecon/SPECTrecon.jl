# rotatez.jl

export imrotate!, imrotate_adj!
export imrotate, imrotate_adj

using LinearInterpolators: LinearSpline
using LinearInterpolators: SparseInterpolator, AffineTransform2D, rotate
using LinearInterpolators: TwoDimensionalTransformInterpolator


"""
    linearinterp!(A, s, e, len)
Assign key values in `SparseInterpolator` (linear) `A`
that are calculated from `s`, `e` and `len`.
`s` means start (x[1])
`e` means end (x[end])
`len` means length (length(x))
"""
function linearinterp!(
    A::SparseInterpolator{<:AbstractFloat},
    s::RealU,
    e::RealU,
    len::Int,
)

    dec = ceil(Int, s) - s
    ncoeff = length(A.C)
    ncol = len
    for i in 1:ncoeff
        if isodd(i)
            A.C[i] = dec
        else
            A.C[i] = 1 - dec
        end
    end

    if e <= ncol
        A.J[end] = ceil(Int, e)
        for i in ncoeff-1:-1:1
            A.J[i] = max(1, A.J[end] - ceil(Int, (ncoeff - i) / 2))
        end
    else
        A.J[1] = floor(Int, s)
        for i in 2:ncoeff
            A.J[i] = min(ncol, A.J[1] + ceil(Int, (i - 1) / 2))
        end
    end
    return A
end


"""
    rotate_x!(output, img, tan_θ, interp)
"""
function rotate_x!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    tan_θ::RealU,
    interp::SparseInterpolator,
)

    len = size(img, 1) # size of a square image
    c_y = (len + 1)/2 # center of yi
    idx = 1:len

    for i in idx
        s = (idx[i] - c_y) * tan_θ + 1
        e = (idx[i] - c_y) * tan_θ + len
        linearinterp!(interp, s, e, len)
        mul!((@view output[:, i]), interp, (@view img[:, i]))
    end
    return output
end


"""
    rotate_x_adj!(output, img, tan_θ, interp)
"""
function rotate_x_adj!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    tan_θ::RealU,
    interp::SparseInterpolator,
)

    len = size(img, 1)
    c_y = (len + 1)/2 # center of yi
    idx = 1:len

    for i in idx
        s = (idx[i] - c_y) * tan_θ + 1
        e = (idx[i] - c_y) * tan_θ + len
        linearinterp!(interp, s, e, len)
        mul!((@view output[:, i]), interp', (@view img[:, i]))
    end
    return output
end


"""
    rotate_y!(output, img, sin_θ, interp)
"""
function rotate_y!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    sin_θ::RealU,
    interp::SparseInterpolator,
)

    len = size(img, 1)
    c_x = (len + 1)/2 # center of xi
    idx = 1:len

    for i in idx
        s = (idx[i] - c_x) * sin_θ + 1
        e = (idx[i] - c_x) * sin_θ + len
        linearinterp!(interp, s, e, len)
        mul!((@view output[i, :]), interp, (@view img[i, :]))
    end
    return output
end


"""
    rotate_y_adj!(output, img, sin_θ, interp)
"""
function rotate_y_adj!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    sin_θ::RealU,
    interp::SparseInterpolator,
)

    len = size(img, 1)
    c_x = (len + 1)/2 # center of xi
    idx = 1:len

    for i in idx
        s = (idx[i] - c_x) * sin_θ + 1
        e = (idx[i] - c_x) * sin_θ + len
        linearinterp!(interp, s, e, len)
        mul!((@view output[i, :]), interp', (@view img[i, :]))
    end
    return output
end



"""
    rotl90!(B::AbstractMatrix, A::AbstractMatrix)
In place version of `rotl90`, returning rotation of `A` in `B`.
"""
function rotl90!(B::AbstractMatrix, A::AbstractMatrix)
    ind1, ind2 = axes(A)
    n = first(ind2) + last(ind2)
    for i in axes(A, 1), j in ind2
        B[n - j, i] = A[i, j]
    end
    return B
end


"""
    rotr90!(B::AbstractMatrix, A::AbstractMatrix)
In place version of `rotr90`, returning rotation of `A` in `B`.
"""
function rotr90!(B::AbstractMatrix, A::AbstractMatrix)
    ind1, ind2 = axes(A)
    m = first(ind1) + last(ind1)
    for i in ind1, j in axes(A, 2)
        B[j, m - i] = A[i, j]
    end
    return B
end


"""
    rot180!(B::AbstractMatrix, A::AbstractMatrix)
In place version of `rot180`, returning rotation of `A` in `B`.
"""
function rot180!(B::AbstractMatrix, A::AbstractMatrix)
    ind1, ind2 = axes(A,1), axes(A,2)
    m, n = first(ind1)+last(ind1), first(ind2)+last(ind2)
    for j in ind2, i in ind1
        B[m-i,n-j] = A[i, j]
    end
    return B
end


"""
    rot_f90!(output, img, m)
In-place version of rotating an image by 90/180/270 degrees
"""
function rot_f90!(output::AbstractMatrix, img::AbstractMatrix, m::Int)
    if m == 0
        output .= img
    elseif m == 1
        rotl90!(output, img)
    elseif m == 2
        rot180!(output, img)
    elseif m == 3
        rotr90!(output, img)
    else
        throw("invalid m!")
    end
    return output
end


"""
    rot_f90_adj!(output, img, m)
    The adjoint of rotating an image by 90/180/270 degrees
"""
function rot_f90_adj!(output::AbstractMatrix, img::AbstractMatrix, m::Int)
    if m == 0
        output .= img
    elseif m == 1
        rotr90!(output, img)
    elseif m == 2
        rot180!(output, img)
    elseif m == 3
        rotl90!(output, img)
    else
        throw("invalid m!")
    end
    return output
end



"""
    imrotate!(output, img, θ, plan)
Rotate a 2D image `img` by angle `θ ∈ [0,2π]`
in counter-clockwise direction (opposite to `imrotate` in Julia)
using 3-pass 1d linear interpolations.
"""
function imrotate!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    θ::RealU,
    plan::PlanRotate{<:Number,RotateMode{:one}},
)
    @boundscheck size(output) == size(img) || throw("size")
    @boundscheck size(img, 1) == size(img, 2) || throw("row != col")

    if mod(θ, 2π) ≈ 0
        output .= img
        return output # todo: check!
    end

    m = mod(floor(Int, 0.5 + θ/(π/2)), 4)

    N = size(img, 1) # M = N!

    if θ ≈ m * (π/2)
        padzero!(plan.workmat2, img,
            (plan.padsize, plan.padsize, plan.padsize, plan.padsize))
        rot_f90!(plan.workmat1, plan.workmat2, m)
    else
        mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
        tan_mod_theta = tan(mod_theta / 2)
        sin_mod_theta = - sin(mod_theta)
        padzero!(plan.workmat1, img,
            (plan.padsize, plan.padsize, plan.padsize, plan.padsize))
        rot_f90!(plan.workmat2, plan.workmat1, m)
        rotate_x!(plan.workmat1, plan.workmat2, tan_mod_theta, plan.interp)
        rotate_y!(plan.workmat2, plan.workmat1, sin_mod_theta, plan.interp)
        rotate_x!(plan.workmat1, plan.workmat2, tan_mod_theta, plan.interp)
    end

    output .= (@view plan.workmat1[plan.padsize .+ (1:N),
                                   plan.padsize .+ (1:N)])

    return output
end


"""
    imrotate_adj!(output, img, θ, plan)
Adjoint of `imrotate!` for a 2D image
using 3-pass 1d linear interpolations.
"""
function imrotate_adj!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    θ::RealU,
    plan::PlanRotate{<:Number,RotateMode{:one}},
)

    @boundscheck size(output) == size(img) || throw("size")
    @boundscheck size(img, 1) == size(img, 2) || throw("row != col")

    if mod(θ, 2π) ≈ 0
        output .= img
        return output
    end

    m = mod(floor(Int, 0.5 + θ/(π/2)), 4)
    N = size(img, 1) # M = N!

    if θ ≈ m * (π/2)
        padzero!(plan.workmat2, img,
            (plan.padsize, plan.padsize, plan.padsize, plan.padsize))
        rot_f90_adj!(plan.workmat1, plan.workmat2, m)
    else
        mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
        tan_mod_theta = tan(mod_theta / 2)
        sin_mod_theta = - sin(mod_theta)
        padzero!(plan.workmat1, img,
            (plan.padsize, plan.padsize, plan.padsize, plan.padsize))
        rotate_x_adj!(plan.workmat2, plan.workmat1, tan_mod_theta, plan.interp)
        rotate_y_adj!(plan.workmat1, plan.workmat2, sin_mod_theta, plan.interp)
        rotate_x_adj!(plan.workmat2, plan.workmat1, tan_mod_theta, plan.interp)
        rot_f90_adj!(plan.workmat1, plan.workmat2, m) # must be two different arguments
    end
    output .= (@view plan.workmat1[plan.padsize .+ (1:N),
                                   plan.padsize .+ (1:N)])
    return output
end


"""
    imrotate!(output, img, θ, plan)
Rotate a 2D image `img` by angle `θ ∈ [0,2π]`
in counter-clockwise direction (opposite to `imrotate` in Julia)
using 2d bilinear interpolation.
"""
function imrotate!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    θ::RealU,
    plan::PlanRotate{T,RotateMode{:two}},
) where {T}

    @boundscheck size(output) == size(img) || throw("size")
    @boundscheck size(img, 1) == size(img, 2) || throw("row != col")

    if mod(θ, 2π) ≈ 0
        output .= img
        return output
    end

    N = size(img, 1) # M = N!

    ker = LinearSpline(T)
    rows = size(plan.workmat2)
    cols = size(plan.workmat1)
    c = ((1 + rows[1]) / 2, (1 + rows[2]) / 2)
    R = c + rotate(2π - θ, AffineTransform2D{T}() - c)
    A = TwoDimensionalTransformInterpolator(rows, cols, ker, R)

    padzero!(plan.workmat1, img,
        (plan.padsize, plan.padsize, plan.padsize, plan.padsize))
    mul!(plan.workmat2, A, plan.workmat1)

    output .= (@view plan.workmat2[plan.padsize .+ (1:N),
                                   plan.padsize .+ (1:N)])
    return output
end


"""
    imrotate(img, θ; method::Symbol=:two)
Rotate a 2D image `img` by angle `θ ∈ [0,2π]`
in counter-clockwise direction (opposite to `imrotate` in Julia)
using either 2d linear interpolation (for `:two`)
or 3-pass 1D interpolation (for `:one`)
"""
function imrotate(
    img::AbstractMatrix{I},
    θ::RealU;
    method::Symbol=:two,
    T::Type{<:AbstractFloat} = promote_type(I, Float32),
) where {I <: Number}
    output = similar(Matrix{T}, size(img))
    plan = plan_rotate(size(img, 1); T, nthread = 1, method)[1]
    imrotate!(output, img, θ, plan)
    return output
end


"""
    imrotate_adj!(output, img, θ, plan)
Adjoint of `imrotate!` for a 2D image
using 2D bilinear interpolation.
"""
function imrotate_adj!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    θ::RealU,
    plan::PlanRotate{T,RotateMode{:two}},
) where {T}

    @boundscheck size(output) == size(img) || throw("size")
    @boundscheck size(img, 1) == size(img, 2) || throw("row != col")

    N = size(img, 1) # M = N!

    ker = LinearSpline(T)
    rows = size(plan.workmat2)
    cols = size(plan.workmat1)
    c = ((1 + rows[1]) / 2, (1 + rows[2]) / 2)
    R = c + rotate(2π - θ, AffineTransform2D{T}() - c)
    A = TwoDimensionalTransformInterpolator(rows, cols, ker, R)

    padzero!(plan.workmat1, img,
        (plan.padsize, plan.padsize, plan.padsize, plan.padsize))
    mul!(plan.workmat2, A', plan.workmat1) # todo: internals of A' ?

    output .= (@view plan.workmat2[plan.padsize .+ (1:N),
                                   plan.padsize .+ (1:N)])
    return output
end


"""
    imrotate_adj(img, θ; method::Symbol=:two)
Adjoint of rotating a 2D image `img` by angle `θ ∈ [0,2π]`
in counter-clockwise direction (opposite to `imrotate` in Julia)
using either 2d linear interpolations or 3-pass 1D interpolation.
"""
function imrotate_adj(
    img::AbstractMatrix{I},
    θ::RealU;
    method::Symbol=:two,
    T::Type{<:AbstractFloat} = promote_type(I, Float32),
) where {I <: Number}
    output = similar(Matrix{T}, size(img))
    plan = plan_rotate(size(img, 1); T, nthread = 1, method)[1]
    imrotate_adj!(output, img, θ, plan)
    return output
end


function _task3(z, fun::Function, output, image3, θ, plans)
    id = Threads.threadid()
#   1 ≤ id ≤ length(plans) || throw("bug: id=$id nplan=$(length(plans))")
    return fun(
        (@view output[:,:,z]),
        (@view image3[:,:,z]),
        θ,
        plans[id],
    )
end

# rotate image3 using foreach
function _imrotate!(
    fun::Function, # imrotate! or imrotate_adj!
    output::AbstractArray{<:RealU,3},
    image3::AbstractArray{<:RealU,3},
    θ::RealU,
    plans::Vector{<:PlanRotate}, # must be ≥ Threads.nthreads()
    ntasks::Int = Threads.nthreads(), # ≥ 1
)

    size(output) == size(image3) || throw(DimensionMismatch())
    size(image3,1) == plans[1].nx || throw("nx")
    length(plans) == Threads.nthreads() || throw("#threads")

    task = z -> _task3(z, fun, output, image3, θ, plans)
    Threads.foreach(task, foreach_setup(1:size(image3,3)); ntasks)

    return output
end


"""
    imrotate!(output, image3, θ, plans, ntasks=nthreads)
In-place version of rotating a 3D `image3` by `θ`
in counter-clockwise direction (opposite to `imrotate` in Julia)
using `foreach` with `ntasks`.
"""
function imrotate!(
    output::AbstractArray{<:RealU,3},
    image3::AbstractArray{<:RealU,3},
    θ::RealU,
    plans::Vector{<:PlanRotate},
    ntasks::Int = Threads.nthreads(),
)
    return _imrotate!(imrotate!, output, image3, θ, plans, ntasks)
end


"""
    imrotate_adj!(output, image3, θ, plans, ntasks=nthreads)
Adjoint of `imrotate!` using `foreach`.
"""
function imrotate_adj!(
    output::AbstractArray{<:RealU,3},
    image3::AbstractArray{<:RealU,3},
    θ::RealU,
    plans::Vector{<:PlanRotate},
    ntasks::Int = Threads.nthreads(),
)
    return _imrotate!(imrotate_adj!, output, image3, θ, plans, ntasks)
end



# rotate image3 using `Threads.@threads`
function _imrotate!(
    fun::Function, # imrotate! or imrotate_adj!
    output::AbstractArray{<:RealU,3},
    image3::AbstractArray{<:RealU,3},
    θ::RealU,
    plans::Vector{<:PlanRotate}, # must be ≥ Threads.nthreads()
    ::Symbol, # :thread
)

    size(output) == size(image3) || throw(DimensionMismatch())
    size(image3,1) == plans[1].nx || throw("nx")
    length(plans) == Threads.nthreads() || throw("#threads")

    Threads.@threads for z in 1:size(image3,3) # 1:nz
        id = Threads.threadid() # thread id
        fun(
            (@view output[:, :, z]),
            (@view image3[:, :, z]),
            θ,
            plans[id],
        )
    end

    return output
end


"""
    imrotate!(output, image3, θ, plans, :thread)
Mutating version of rotating a 3D `image3` by `θ`
in counter-clockwise direction (opposite of `imrotate` in Julia)
writing result into `output`,
using `Threads.@threads`.
"""
function imrotate!(
    output::AbstractArray{<:RealU,3},
    image3::AbstractArray{<:RealU,3},
    θ::RealU,
    plans::Vector{<:PlanRotate},
    ::Symbol, # :thread
)
    return _imrotate!(imrotate!, output, image3, θ, plans, :thread)
end


"""
    imrotate_adj!(output, image3, θ, plans, :thread)
Adjoint of `imrotate!` using `@threads`.
"""
function imrotate_adj!(
    output::AbstractArray{<:RealU,3},
    image3::AbstractArray{<:RealU,3},
    θ::RealU,
    plans::Vector{<:PlanRotate},
    ::Symbol, # :thread
)
    return _imrotate!(imrotate_adj!, output, image3, θ, plans, :thread)
end
