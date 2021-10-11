# rotatez.jl

export imrotate1!, imrotate1_adj!
export imrotate1, imrotate1_adj
export imrotate2!, imrotate2_adj!
export imrotate2, imrotate2_adj
export imrotate!, imrotate_adj!

using LinearInterpolators: LinearSpline
using LinearInterpolators: SparseInterpolator, AffineTransform2D, rotate
using LinearInterpolators: TwoDimensionalTransformInterpolator


"""
    linearinterp!(A, x)
Assign key values in `SparseInterpolator` (linear) `A` that are calculated from `x`.
`x` must be a constant vector
"""
function linearinterp!(
    A::SparseInterpolator{<:AbstractFloat},
    x::AbstractVector{<:RealU},
)
    # x must be a constant vector
    dec = ceil(Int, x[1]) - x[1]
    ncoeff = length(A.C)
    ncol = length(x)
    for i = 1:ncoeff
        if isodd(i)
            A.C[i] = dec
        else
            A.C[i] = 1 - dec
        end
    end
    if x[end] <= ncol
        A.J[end] = ceil(Int, x[end])
        for i = ncoeff-1:-1:1
            A.J[i] = max(1, A.J[end] - ceil(Int, (ncoeff - i) / 2))
        end
    else
        A.J[1] = floor(Int, x[1])
        for i = 2:ncoeff
            A.J[i] = min(ncol, A.J[1] + ceil(Int, (i - 1) / 2))
        end
    end
    return A
end


"""
    rotate_x!(output, img, tan_θ, workvec, interp)
"""
function rotate_x!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    tan_θ::RealU,
    workvec::AbstractVector{<:RealU},
    interp::SparseInterpolator,
)

    len = length(workvec)
    c_y = (len + 1)/2 # center of yi
    idx = 1:len

    for i in idx
        workvec .= idx[i]
        broadcast!(-, workvec, workvec, c_y)
        broadcast!(*, workvec, workvec, tan_θ)
        broadcast!(+, workvec, workvec, idx)
        linearinterp!(interp, workvec)
        mul!((@view output[:, i]), interp, (@view img[:, i])) # need mul! to avoid allocating
    end
    return output
end


"""
    rotate_x_adj!(output, img, tan_θ, workvec, interp)
"""
function rotate_x_adj!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    tan_θ::RealU,
    workvec::AbstractVector{<:RealU},
    interp::SparseInterpolator,
)

    len = length(workvec)
    c_y = (len + 1)/2 # center of yi
    idx = 1:len

    for i in idx
        workvec .= idx[i]
        broadcast!(-, workvec, workvec, c_y)
        broadcast!(*, workvec, workvec, tan_θ)
        broadcast!(+, workvec, workvec, idx)
        linearinterp!(interp, workvec)
        mul!((@view output[:, i]), interp', (@view img[:, i])) # need mul! to avoid allocating
    end
    return output
end


"""
    rotate_y!(output, img, sin_θ, workvec, interp)
"""
function rotate_y!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    sin_θ::RealU,
    workvec::AbstractVector{<:RealU},
    interp::SparseInterpolator,
)

    len = length(workvec)
    c_x = (len + 1)/2 # center of xi
    idx = 1:len

    for i in idx
        workvec .= idx[i]
        broadcast!(-, workvec, workvec, c_x)
        broadcast!(*, workvec, workvec, sin_θ)
        broadcast!(+, workvec, workvec, idx)
        linearinterp!(interp, workvec)
        mul!((@view output[i, :]), interp, (@view img[i, :]))
    end
    return output

end


"""
    rotate_y_adj!(output, img, sin_θ, workvec, interp)
"""
function rotate_y_adj!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    sin_θ::RealU,
    workvec::AbstractVector{<:RealU},
    interp::SparseInterpolator,
)

    len = length(workvec)
    c_x = (len + 1)/2 # center of xi
    idx = 1:len

    for i in idx
        workvec .= idx[i]
        broadcast!(-, workvec, workvec, c_x)
        broadcast!(*, workvec, workvec, sin_θ)
        broadcast!(+, workvec, workvec, idx)
        linearinterp!(interp, workvec)
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
    for i = axes(A, 1), j = ind2
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
    for i = ind1, j = axes(A, 2)
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
    for j=ind2, i=ind1
        B[m-i,n-j] = A[i, j]
    end
    return B
end


"""
    rot_f90!(output, img, m)
In-place version of rotating an image by 90/180/270 degrees
"""
function rot_f90!(
    output::AbstractMatrix,
    img::AbstractMatrix,
    m::Int,
)
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
function rot_f90_adj!(
    output::AbstractMatrix,
    img::AbstractMatrix,
    m::Int,
)
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
    imrotate1!(output, img, θ, plan)
Rotate an image by angle θ (must be within 0 to 2π) in counter-clockwise direction
(opposite to `imrotate` in Julia) using a series of 1d linear interpolations.
"""
function imrotate1!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    θ::RealU,
    plan::PlanRotate,
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
        padzero!(plan.workmat2, img, (plan.padsize, plan.padsize, plan.padsize, plan.padsize))
        rot_f90!(plan.workmat1, plan.workmat2, m)
    else
        mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
        tan_mod_theta = tan(mod_theta / 2)
        sin_mod_theta = - sin(mod_theta)
        padzero!(plan.workmat1, img, (plan.padsize, plan.padsize, plan.padsize, plan.padsize))
        rot_f90!(plan.workmat2, plan.workmat1, m)
        rotate_x!(plan.workmat1, plan.workmat2, tan_mod_theta, plan.workvec, plan.interp)
        rotate_y!(plan.workmat2, plan.workmat1, sin_mod_theta, plan.workvec, plan.interp)
        rotate_x!(plan.workmat1, plan.workmat2, tan_mod_theta, plan.workvec, plan.interp)
    end

    output .= (@view plan.workmat1[plan.padsize + 1 : plan.padsize + N,
                                   plan.padsize + 1 : plan.padsize + N])

    return output
end


"""
    imrotate1(img, θ)
Rotate an image by angle θ (must be within 0 to 2π) in counter-clockwise direction
(opposite to `imrotate` in Julia) using a series of 1d linear interpolations.
"""
function imrotate1(img::AbstractMatrix{<:RealU}, θ::RealU)
    output = similar(img)
    plan = plan_rotate(size(img, 1); T = eltype(img), nthread = 1)[1]
    imrotate1!(output, img, θ, plan)
    return output
end


"""
    imrotate1_adj!(output, img, θ, plan)
The adjoint of rotating an image by angle θ (must be within 0 to 2π) in counter-clockwise direction
(opposite to `imrotate` in Julia) using a series of 1d linear interpolations.
"""
function imrotate1_adj!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    θ::RealU,
    plan::PlanRotate,
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
        padzero!(plan.workmat2, img, (plan.padsize, plan.padsize, plan.padsize, plan.padsize))
        rot_f90_adj!(plan.workmat1, plan.workmat2, m)
    else
        mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
        tan_mod_theta = tan(mod_theta / 2)
        sin_mod_theta = - sin(mod_theta)
        padzero!(plan.workmat1, img, (plan.padsize, plan.padsize, plan.padsize, plan.padsize))
        rotate_x_adj!(plan.workmat2, plan.workmat1, tan_mod_theta, plan.workvec, plan.interp)
        rotate_y_adj!(plan.workmat1, plan.workmat2, sin_mod_theta, plan.workvec, plan.interp)
        rotate_x_adj!(plan.workmat2, plan.workmat1, tan_mod_theta, plan.workvec, plan.interp)
        rot_f90_adj!(plan.workmat1, plan.workmat2, m) # must be two different arguments
    end
    output .= (@view plan.workmat1[plan.padsize + 1 : plan.padsize + N,
                                   plan.padsize + 1 : plan.padsize + N])
    return output
end


"""
    imrotate1_adj(img, θ)
The adjoint of rotating an image by angle θ (must be within 0 to 2π) in counter-clockwise direction
(opposite to `imrotate` in Julia) using a series of 1d linear interpolations.
"""
function imrotate1_adj(img::AbstractMatrix{<:RealU}, θ::RealU)
    output = similar(img)
    plan = plan_rotate(size(img, 1); T = eltype(img), nthread = 1)[1]
    imrotate1_adj!(output, img, θ, plan)
    return output
end


"""
    imrotate2!(output, img, θ, plan)
Rotate an image by angle θ (must be within 0 to 2π) in counter-clockwise direction
(opposite to `imrotate` in Julia) using 2d linear interpolations.
"""
function imrotate2!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    θ::RealU,
    plan::PlanRotate,
)

    @boundscheck size(output) == size(img) || throw("size")
    @boundscheck size(img, 1) == size(img, 2) || throw("row != col")

    if mod(θ, 2π) ≈ 0
        output .= img
        return output
    end

    N = size(img, 1) # M = N!

    T = eltype(img)
    ker = LinearSpline(T)
    rows = size(plan.workmat2)
    cols = size(plan.workmat1)
    c = ((1 + rows[1]) / 2, (1 + rows[2]) / 2)
    R = c + rotate(2π - θ, AffineTransform2D{T}() - c)
    A = TwoDimensionalTransformInterpolator(rows, cols, ker, R)

    padzero!(plan.workmat1, img, (plan.padsize, plan.padsize, plan.padsize, plan.padsize))
    mul!(plan.workmat2, A, plan.workmat1)

    output .= (@view plan.workmat2[plan.padsize + 1 : plan.padsize + N,
                                   plan.padsize + 1 : plan.padsize + N])
    return output
end


"""
    imrotate2(img, θ)
Rotate an image by angle θ (must be within 0 to 2π) in counter-clockwise direction
(opposite to `imrotate` in Julia) using 2d linear interpolations.
"""
function imrotate2(img::AbstractMatrix{<:RealU}, θ::RealU)
    output = similar(img)
    plan = plan_rotate(size(img, 1); T = eltype(img), nthread = 1)[1]
    imrotate2!(output, img, θ, plan)
    return output
end


"""
    imrotate2_adj!(output, img, θ, plan)
The adjoint of rotating an image by angle θ (must be within 0 to 2π) in counter-clockwise direction
(opposite to `imrotate` in Julia) using 2d linear interpolations.
"""
function imrotate2_adj!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    θ::RealU,
    plan::PlanRotate,
)

    @boundscheck size(output) == size(img) || throw("size")
    @boundscheck size(img, 1) == size(img, 2) || throw("row != col")

    N = size(img, 1) # M = N!

    T = eltype(img)
    ker = LinearSpline(T)
    rows = size(plan.workmat2)
    cols = size(plan.workmat1)
    c = ((1 + rows[1]) / 2, (1 + rows[2]) / 2)
    R = c + rotate(2π - θ, AffineTransform2D{T}() - c)
    A = TwoDimensionalTransformInterpolator(rows, cols, ker, R)

    padzero!(plan.workmat1, img, (plan.padsize, plan.padsize, plan.padsize, plan.padsize))
    mul!(plan.workmat2, A', plan.workmat1)

    output .= (@view plan.workmat2[plan.padsize + 1 : plan.padsize + N,
                                   plan.padsize + 1 : plan.padsize + N])
    return output
end


"""
    imrotate2_adj(img, θ)
The adjoint of rotating an image by angle θ (must be within 0 to 2π) in counter-clockwise direction
(opposite to `imrotate` in Julia) using 2d linear interpolations.
"""
function imrotate2_adj(img::AbstractMatrix{<:RealU}, θ::RealU)
    output = similar(img)
    plan = plan_rotate(size(img, 1); T = eltype(img), nthread = 1)[1]
    imrotate2_adj!(output, img, θ, plan)
    return output
end


"""
    imrotate!(output, img, θ, plan)
In-place version of rotating an `image` by `θ` in counter-clockwise direction
(opposite to `imrotate` in Julia)
"""
function imrotate!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    θ::RealU,
    plan::PlanRotate,
)

    if plan.method === :one
        imrotate1!(output, img, θ, plan)
    else
        imrotate2!(output, img, θ, plan)
    end
    return output
end


"""
    imrotate_adj!(output, img, θ, plan)
In-place version of the adjoint of rotating an `image` by `θ` in counter-clockwise direction
(opposite to `imrotate` in Julia)
"""
function imrotate_adj!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    θ::RealU,
    plan::PlanRotate,
)

    if plan.method === :one
        imrotate1_adj!(output, img, θ, plan)
    else
        imrotate2_adj!(output, img, θ, plan)
    end
    return output
end


# prepare for "foreach" threaded computation
_setup = (z) -> Channel() do ch
    foreach(i -> put!(ch, i), z)
end


"""
    imrotate!(output, image3, θ, plans)
In-place version of rotating a 3D `image3` by `θ`
in counter-clockwise direction (opposite to `imrotate` in Julia)
"""
function imrotate!(
    output::AbstractArray{<:RealU,3},
    image3::AbstractArray{<:RealU,3},
    θ::RealU,
    plans::Vector{<:PlanRotate},
)

    size(output) == size(image3) || throw(DimensionMismatch())

    fun = z -> imrotate!(
        (@view output[:,:,z]),
        (@view image3[:,:,z]),
        θ,
        plans[Threads.threadid()],
    )

    ntasks = length(plans)
    Threads.foreach(fun, _setup(1:size(image3,3)); ntasks)

#=
    Threads.@threads for z = 1:size(image3,3) # 1:nz
        id = Threads.threadid() # thread id

        imrotate!(
            (@view output[:,:,z]),
            (@view image3[:,:,z]),
            θ,
            plans[id],
        )
    end
=#

    return output
end


"""
    imrotate_adj!(output, image3, θ, plan)
In-place version of the adjoint of rotating a 3D `image3` by `θ`
in counter-clockwise direction (opposite to `imrotate` in Julia)
"""
function imrotate_adj!(
    output::AbstractArray{<:RealU,3},
    image3::AbstractArray{<:RealU,3},
    θ::RealU,
    plans::Vector{<:PlanRotate},
)

    size(output) == size(image3) || throw(DimensionMismatch())

    fun = z -> imrotate_adj!(
        (@view output[:,:,z]),
        (@view image3[:,:,z]),
        θ,
        plans[Threads.threadid()],
    )

    ntasks = length(plans)
    Threads.foreach(fun, _setup(1:size(image3,3)); ntasks)

    return output
end
