# spectplan.jl

export SPECTplan

"""
    SPECTplan
Struct for storing key factors for a SPECT system model
- `T` datatype of work arrays
- `imgsize` size of image
- `px,pz` psf dimension
- `imgr [nx, ny, nz]` 3D rotated version of image
- `add_img [nx, ny, nz]` 3D image for adding views and backprojection
- `mumap [nx,ny,nz]` attenuation map, must be 3D, possibly zeros()
- `mumapr [nx, ny, nz]` 3D rotated mumap
- `exp_mumapr [nx, nz]` 2D exponential rotated mumap
- `psfs [px,pz,ny,nview]` point spread function, must be 4D, with `px and `pz` odd, and symmetric for each slice
- `nview` number of views, must be integer
- `viewangle` set of view angles, must be from 0 to 2π
- `interpmeth` interpolation method: `:one` means 1d; `:two` means 2d
- `mode` pre-allocation method: `:fast` means faster; `:mem` means use less memory
- `dy` voxel size in y direction (`dx` is the same value)
- `nthread` number of CPU threads used to process data; must be integer
- `planrot` Vector of struct `PlanRotate`
- `planpsf` Vector of struct `PlanPSF`
Currently code assumes the following:
* each of the `nview` projection views is `[nx,nz]`
* `nx = ny`
* uniform angular sampling
* `psf` is symmetric
* multiprocessing using # of threads specified by `Threads.nthreads()`
"""
struct SPECTplan{T}
    T::Type{<:AbstractFloat} # default type for work arrays etc.
    imgsize::NTuple{3, Int}
    px::Int
    pz::Int
    imgr::Union{Array{T, 3}, Vector{Array{T, 3}}} # 3D rotated image, (nx, ny, nz)
    add_img::Union{Array{T, 3}, Vector{Array{T, 3}}}
    mumap::Array{T, 3} # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    mumapr::Union{Array{T, 3}, Vector{Array{T, 3}}} # 3D rotated mumap, (nx, ny, nz)
    exp_mumapr::Vector{Matrix{T}} # 2D exponential rotated mumap, (nx, ny)
    psfs::Array{T, 4} # PSFs must be 4D, [px, pz, ny, nview], finally be centered psf
    nview::Int # number of views
    viewangle::StepRangeLen{T}
    interpmeth::Symbol
    mode::Symbol
    dy::T
    nthread::Int # number of threads
    planrot::Vector{PlanRotate}
    planpsf::Vector{PlanPSF}

    """
        SPECTplan(mumap, psfs, dy; T, viewangle, interpmeth, nthread, mode)
    """
    function SPECTplan(
        mumap::Array{<:RealU, 3},
        psfs::Array{<:RealU, 4},
        dy::RealU;
        T::Type{<:AbstractFloat} = promote_type(eltype(mumap), Float32),
        viewangle::StepRangeLen{<:RealU} = (0:size(psfs, 4) - 1) / size(psfs, 4) * T(2π), # set of view angles
        interpmeth::Symbol = :two, # :one is for 1d interpolation, :two is for 2d interpolation
        nthread::Int = Threads.nthreads(),
        mode::Symbol = :fast,
    )

        # convert to the same type
        dy = convert(T, dy)
        mumap .= T.(mumap)
        psfs .= T.(psfs)

        (nx, ny, nz) = size(mumap) # typically 128 x 128 x 81

        isequal(nx, ny) || throw("nx != ny")
        (iseven(nx) && iseven(ny)) || throw("nx odd")

        imgsize = (nx, ny, nz)
        # check psf
        px, pz, _, nview = size(psfs)
        (isodd(px) && isodd(pz)) || throw("non-odd size psfs")
        all(mapslices(x -> x == reverse(x, dims=:), psfs, dims = [1, 2])) ||
            throw("asym. psf")

        # check interpidx
        (interpmeth === :one || interpmeth === :two) || throw("bad interpmeth")

        # remember to check if nthread == Threads.nthreads()
        (nthread == Threads.nthreads()) || throw("bad nthread")

        # check mode
        (mode === :fast || mode === :mem) || throw("bad mode")

        if mode === :fast
            # imgr stores 3D image in different view angles
            imgr = [Array{T, 3}(undef, nx, ny, nz) for id in 1:nthread]
            # add_img stores 3d image for backprojection
            add_img = [Array{T, 3}(undef, nx, ny, nz) for id in 1:nthread]
            # mumapr stores 3D mumap in different view angles
            mumapr = [Array{T, 3}(undef, nx, ny, nz) for id in 1:nthread]
        else
            # imgr stores 3D image in different view angles
            imgr = Array{T, 3}(undef, nx, ny, nz)
            # add_img stores 3d image for backprojection
            add_img = Array{T ,3}(undef, nx, ny, nz)
            # mumapr stores 3D mumap in different view angles
            mumapr = Array{T, 3}(undef, nx, ny, nz)
        end

        exp_mumapr = [Matrix{T}(undef, nx, nz) for id in 1:nthread]

        planrot = plan_rotate(nx; T, method = interpmeth)

        planpsf = plan_psf(; nx, nz, px, pz, nthread, T)

        new{T}(T, # default type for work arrays etc.
            imgsize,
            px,
            pz,
            imgr, # 3D rotated image, (nx, ny, nz)
            add_img,
            mumap, # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
            mumapr, # 3D rotated mumap, (nx, ny, nz)
            exp_mumapr,
            psfs, # PSFs must be 4D, [px, pz, ny, nview], finally be centered psf
            nview, # number of views
            viewangle,
            interpmeth,
            mode,
            dy,
            nthread, # number of threads
            planrot,
            planpsf,
        )
    end
end


"""
    show(io::IO, ::MIME"text/plain", plan::SPECTplan)
"""
function Base.show(io::IO, ::MIME"text/plain", plan::SPECTplan{T}) where {T}
    t = typeof(plan)
    println(io, t)
    for f in (:imgsize, :px, :pz, :nview, :viewangle, :interpmeth, :mode, :dy, :nthread)
        p = getfield(plan, f)
        t = typeof(p)
        println(io, " ", f, "::", t, " ", p)
    end
    for f in (:mumap, )
        p = getfield(plan, f)
        println(io, " ", f, ":", " ", summary(p))
    end
    println(io, " (", sizeof(plan), " bytes)")
end


"""
    sizeof(::SPECTplan)
Show size in bytes of `SPECTplan` object.
"""
function Base.sizeof(ob::T) where {T <: Union{SPECTplan}}
    sum(f -> sizeof(getfield(ob, f)), fieldnames(typeof(ob)))
end
