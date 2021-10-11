# plan-rotate.jl

export PlanRotate, plan_rotate
using LinearInterpolators: SparseInterpolator, LinearSpline


"""
    PlanRotate(nx::Int; T::DataType, method::Symbol)
Struct for storing work arrays and factors for 2D square image rotation for one thread
- `T` datatype of work arrays
- `nx` must be Int
- `method` interpolation methods, `:one` is to use 3-pass 1D interpolation, `:two` is to use 2D interpolation
- `padsize` padsize, must be Int
- `workmat1 [nx + 2 * padsize, nx + 2 * padsize]` padded work matrix
- `workmat2 [nx + 2 * padsize, nx + 2 * padsize]` padded work matrix
- `workvec [nx + 2 * padsize, ]` padded work vector
- `interp` Sparse [linear] interpolator
"""
struct PlanRotate{T}
    nx::Int
    method::Symbol
    padsize::Int
    workmat1::Matrix{T}
    workmat2::Matrix{T}
    workvec::Vector{T}
    interp::SparseInterpolator{T, 2, 1}

    function PlanRotate(
        nx::Int ;
        T::DataType = Float32,
        method::Symbol = :two, # :one is for 1d interpolation, :two is for 2d interpolation
    )

        @assert (method == :one || method == :two) || throw("bad method") # check interp method

        # only support the case that the image is square

        padsize = ceil(Int, 1 + nx * sqrt(2)/2 - nx / 2)

        # for both 3-pass 1D and 2D interpolators:
        workmat1 = Matrix{T}(undef, nx + 2 * padsize, nx + 2 * padsize)
        workmat2 = Matrix{T}(undef, nx + 2 * padsize, nx + 2 * padsize)

        # used for 1D interpolation only
        # todo: look for non-allocating 1D interpolator?
#       workvec_x = Vector{T}(undef, nx+padu_rot+padd_rot) # todo: why won't work?
#       workvec_y = Vector{T}(undef, ny+padl_rot+padr_rot)
        workvec = zeros(T, nx + 2 * padsize) # cannot initialize as undef, otherwise interp cannot be properly initialized

        interp = SparseInterpolator(LinearSpline(T), workvec, length(workvec))

        S = typeof(interp)

        new{T}(nx,
            method,
            padsize,
            workmat1,
            workmat2,
            workvec,
            interp,
        )
    end
end


"""
    plan_rotate(nx::Int; nthread::Int, T::DataType, method::Symbol)
Make `Vector` of `PlanRotate` structs
for storing work arrays and factors for 2D square image rotation.

# Input
- `nx::Int` must equal to `ny`
# Option
- `T` : datatype of work arrays, defaults to `Float32`
- `method::Symbol` : default is `:two` for 2D interpolation;
  use `:one` for 3-pass rotation with 1D interpolation
- `nthread::Int` # of threads, defaults to `Threads.nthreads()`
  warning: must use that default currently!
"""
function plan_rotate(
    nx::Int ;
    nthread::Int = Threads.nthreads(),
    T::DataType = Float32,
    method::Symbol = :two,
    )
    return [PlanRotate(nx; T, method) for id = 1:nthread]
end


"""
    show(io::IO, ::MIME"text/plain", plan::PlanRotate)
"""
function Base.show(io::IO, ::MIME"text/plain", plan::PlanRotate{T}) where T
    t = typeof(plan)
    println(io, t)
    for f in (:nx, :method, :padsize)
        p = getproperty(plan, f)
        t = typeof(p)
        println(io, " ", f, "::", t, " ", p)
    end
    for f in (:interp, :workmat1, :workmat2, :workvec)
        p = getproperty(plan, f)
        println(io, " ", f, ":", " ", summary(p))
    end
    println(io, " (", sizeof(plan), " bytes)")
end


"""
    show(io::IO, mime::MIME"text/plain", vp::Vector{<:PlanRotate})
"""
function Base.show(io::IO, mime::MIME"text/plain", vp::Vector{<:PlanRotate})
    t = typeof(vp)
    println(io, length(vp), "-element ", t)
#   show(io, mime, vp[1])
end


"""
    sizeof(::PlanRotate)
Show size in bytes of `PlanRotate` object.
"""
function Base.sizeof(ob::T) where {T <: Union{PlanRotate, SparseInterpolator}}
    sum(f -> sizeof(getfield(ob, f)), fieldnames(typeof(ob)))
end
