# spectplan.jl

export SPECTplan

"""
    SPECTplan
Struct for storing key factors for a SPECT system model
- `T` datatype of work arrays
- `imgsize` size of image
- `nx_psf` first dimension of psf
- `imgr [nx, ny, nz]` 3D rotated version of image
- `add_img [nx, ny, nz]` 3D image for adding views and backprojection
- `mumap [nx,ny,nz]` attenuation map, must be 3D, possibly zeros()
- `mumapr [nx, ny, nz]` 3D rotated mumap
- `exp_mumapr [nx, nz]` 2D exponential rotated mumap
- `psfs [nx_psf,nz_psf,ny,nview]` point spread function, must be 4D, with `nx_psf` and `nz_psf` odd, and symmetric for each slice
- `nview` number of views, must be integer
- `viewangle` set of view angles, must be from 0 to 2π
- `interpmeth` interpolation method, :one means 1d, :two means 2d
- `mode` pre-allcoation method, :fast means faster, :mem means use less memory
- `dy` voxel size in y direction (dx is the same value)
- `nthread` number of CPU threads used to process data, must be integer
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
    T::DataType # default type for work arrays etc.
    imgsize::NTuple{3, Int}
    nx_psf::Int
    imgr::Union{Array{T, 3}, Vector{Array{T, 3}}} # 3D rotated image, (nx, ny, nz)
    add_img::Union{Array{T, 3}, Vector{Array{T, 3}}}
    mumap::Array{T, 3} # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    mumapr::Union{Array{T, 3}, Vector{Array{T, 3}}} # 3D rotated mumap, (nx, ny, nz)
    exp_mumapr::Vector{Matrix{T}} # 2D exponential rotated mumap, (nx, ny)
    psfs::Array{T, 4} # PSFs must be 4D, [nx_psf, nz_psf, ny, nview], finally be centered psf
    nview::Int # number of views
    viewangle::StepRangeLen{T}
    interpmeth::Symbol
    mode::Symbol
    dy::T
    nthread::Int # number of threads
    planrot::Vector{PlanRotate}
    planpsf::Vector{PlanPSF}
    # other options for how to do the projection?

    function SPECTplan(mumap::Array{<:RealU, 3},
                       psfs::Array{<:RealU, 4},
                       dy::RealU;
                       T::DataType = promote_type(eltype(mumap), Float32),
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
        nx_psf = size(psfs, 1)
        nz_psf = size(psfs, 2)
        nview = size(psfs, 4)
        (isodd(nx_psf) && isodd(nz_psf)) || throw("non-odd size psfs")
        all(mapslices(x -> x == transpose(x), psfs, dims = [1, 2])) || throw("asym. psf tran.")
        all(mapslices(x -> x == reverse(x), psfs, dims = [1, 2])) || throw("asym. psf rever.")

        # check interpidx
        (interpmeth === :one || interpmeth === :two) || throw("bad interpmeth")

        # remember to check if nthread == Threads.nthreads()
        (nthread == Threads.nthreads()) || throw("bad nthread")

        # check mode
        (mode === :fast || mode === :mem) || throw("bad mode")

        if mode === :fast
            # imgr stores 3D image in different view angles
            imgr = [Array{T, 3}(undef, nx, ny, nz) for id = 1:nthread]
            # add_img stores 3d image for backprojection
            add_img = [Array{T, 3}(undef, nx, ny, nz) for id = 1:nthread]
            # mumapr stores 3D mumap in different view angles
            mumapr = [Array{T, 3}(undef, nx, ny, nz) for id = 1:nthread]
        else
            # imgr stores 3D image in different view angles
            imgr = Array{T, 3}(undef, nx, ny, nz)
            # add_img stores 3d image for backprojection
            add_img = Array{T ,3}(undef, nx, ny, nz)
            # mumapr stores 3D mumap in different view angles
            mumapr = Array{T, 3}(undef, nx, ny, nz)
        end

        exp_mumapr = [Matrix{T}(undef, nx, nz) for id = 1:nthread]

        planrot = plan_rotate(nx; nthread = nthread, T = T, method = interpmeth)

        planpsf = plan_psf(nx, nz, nx_psf; nthread = nthread, T = T)

        new{T}(T, # default type for work arrays etc.
               imgsize,
               nx_psf,
               imgr, # 3D rotated image, (nx, ny, nz)
               add_img,
               mumap, # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
               mumapr, # 3D rotated mumap, (nx, ny, nz)
               exp_mumapr,
               psfs, # PSFs must be 4D, [nx_psf, nz_psf, ny, nview], finally be centered psf
               nview, # number of views
               viewangle,
               interpmeth,
               mode,
               dy,
               nthread, # number of threads
               planrot,
               planpsf,
               )
         # creates objects of the block's type (inner constructor methods).
    end
end
