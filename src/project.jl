# project.jl

"""
    SPECTplan
Struct for storing key factors for a SPECT system model
- `mumap [nx,ny,nz]` attenuation map, must be 3D, possibly zeros()
- `psfs [nx_psf,nz_psf,ny,nview]` point spread function, must be 4D, with `nx_psf` and `nz_psf` odd, and symmetric for each slice
- `nview` number of views, must be integer
- `ncore` number of cores, must be integer
- `rotateforw!` forward rotation operation, default is using 1d linear interpolation
- `rotateadjt!` adjoint rotation operation, default is using 1d linear interpolation
- `viewangle` set of view angles, must be from 0 to 2π
- `dy` voxel size in y direction (dx is the same value)
- `nx` number of voxels in x direction of the image, must be integer
- `ny` number of voxels in y direction of the image, must be integer
- `nz` number of voxels in z direction of the image, must be integer
- `nx_psf` number of voxels in x direction of the psf, must be integer
- `nz_psf` number of voxels in z direction of the psf, must be integer
- `padrepl` padding function using replicate border condition
- `padzero` padding function using zero border condition
- {padleft,padright,padup,paddown} pixels padded along {left,right,up,down} direction for convolution with psfs, must be integer
- `pad_rotate_x` padded pixels for rotating along x axis, must be integer
- `pad_rotate_y` padded pixels for rotating along y axis, must be integer
- `alg` algorithms used for convolution, default is FFT
- `imgr` 3D rotated image
- `mumapr` 3D rotated mumap
- `ncore_iter_y` # of outer iterations through ny using multi-processing, must be integer
- `ncore_array_y` array that stores how many cores are used when iterating through ny
- `ncore_iter_z` # of outer iterations through nz using multi-processing, must be integer
- `ncore_array_z` array that stores how many cores are used when iterating through nz

Currently code assumes each of the `nview` projection views is `[nx,nz]`
Currently code assumes `nx = ny`
Currently code assumes uniform angular sampling
Currently imrotate3 code only supports linear interpolation
Currently code uses multiprocessing using # of cores specified by Threads.nthreads() in Julia
"""
struct SPECTplan
    mumap::AbstractArray{<:Real, 3} # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:Real, 4} # PSFs must be 4D, [nx_psf, nz_psf, ny, nview], finally be centered psf
    nview::Int # number of views
    ncore::Int # number of cores
    rotateforw!::Function
    rotateadjt!::Function
    viewangle::AbstractVector
    dy::Float32
    nx::Int
    ny::Int
    nz::Int
    nx_psf::Int
    nz_psf::Int
    padrepl::Function
    padzero::Function
    padleft::Int
    padright::Int
    padup::Int
    paddown::Int
    pad_rotate_x::Int
    pad_rotate_y::Int
    alg::Any
    imgr::AbstractArray{<:Real, 3} # 3D rotated image, (nx, ny, nz)
    mumapr::AbstractArray{<:Real, 3} # 3D rotated mumap, (nx, ny, nz)
    ncore_iter_y::Int
    ncore_array_y::AbstractVector
    ncore_iter_z::Int
    ncore_array_z::AbstractVector

    # other options for how to do the projection?
    function SPECTplan(mumap::AbstractArray{<:Real,3},
                        psfs::AbstractArray{<:Real,4},
                        nview::Int,
                        dy::RealU ;
                        viewangle::AbstractVector = (0:nview - 1) / nview * (2π), # set of view angles
                        interpidx::Int = 1, # 1 is for 1d interpolation, 2 is for 2d interpolation
                        conv_alg::Symbol = :fft, # convolution algorithms, default is fft
                        padleft::Int = _padleft(mumap, psfs),
                        padright::Int = _padright(mumap, psfs),
                        padup::Int = _padup(mumap, psfs),
                        paddown::Int = _paddown(mumap, psfs))
        # check nx = ny ? typically 128 x 128 x 81
        nx, ny, nz = size(mumap)
        nx_psf = size(psfs, 1)
        nz_psf = size(psfs, 2)
        @assert isequal(nx, ny)
        @assert iseven(nx) && iseven(ny)
        @assert isodd(nx_psf) && isodd(nz_psf)
        @assert all(mapslices(x -> x == reverse(x), psfs, dims = [1, 2]))
        # center the psfs
        psfs = OffsetArray(psfs, OffsetArrays.Origin(-Int((nx_psf-1)/2), -Int((nz_psf-1)/2), 1, 1))
        # todo: needs a rotation plan here
        if interpidx == 1
            rotateforw! = imrotate3!
            rotateadjt! = imrotate3_adj!
        elseif interpidx == 2
            rotateforw! = imrotate3emmt!
            rotateadjt! = imrotate3emmt_adj!
        else
            throw("invalid interpidx!")
        end

        pad_rotate_x = ceil(Int, 1 + nx * sqrt(2)/2 - nx / 2)
        pad_rotate_y = ceil(Int, 1 + ny * sqrt(2)/2 - ny / 2)
        # mypad1 = x -> padarray(x, Pad(:replicate, (nx_psf, nz_psf), (nx_psf, nz_psf)))
        padrepl = x -> BorderArray(x, Pad(:replicate, (padleft, padup), (padright, paddown)))
        padzero = x -> BorderArray(x, Fill(0, (padleft, padup), (padright, paddown)))

        ncore = Threads.nthreads()

        # allocate working buffers:
        # imgr stores 3D image in different view angles
        imgr = zeros(promote_type(eltype(mumap), Float32), nx, ny, nz)
        # mumapr stores 3D mumap in different view angles
        mumapr = zeros(promote_type(eltype(mumap), Float32), nx, ny, nz)

        if conv_alg === :fft
            alg = Algorithm.FFT()
        elseif conv_alg === :fir
            alg = Algorithm.FIR()
        else
            throw("invalid convolution algorithm choice!")
        end

        ncore_iter_y = ceil(Int, ny / ncore) # 16
        ncore_array_y = ncore * ones(Int, ncore_iter_y) # [8, 8, 8, ..., 8]
        ncore_array_y[end] = ny - (ncore_iter_y - 1) * ncore # [8, 8, 8, ..., 8]

        ncore_iter_z = ceil(Int, nz / ncore) # 12
        ncore_array_z = ncore * ones(Int, ncore_iter_z) # [8, 8, 8, ..., 8]
        ncore_array_z[end] = nz - (ncore_iter_z - 1) * ncore # [8, 8, 8, ..., 1]

        new(mumap, psfs, nview, ncore, rotateforw!, rotateadjt!, viewangle,
            dy, nx, ny, nz, nx_psf, nz_psf, padrepl, padzero,
            padleft, padright, padup, paddown, pad_rotate_x, pad_rotate_y,
            alg, imgr, mumapr, ncore_iter_y, ncore_array_y, ncore_iter_z, ncore_array_z)
        #  creates objects of the block's type (inner constructor methods).
    end
end

"""
    Workarray_s
    Struct for storing keys of the work array for a single thread
- `padimg` 2D padded image for imfilter
- `pad_imgr` padded 2D rotated image
- `pad_imgr_tmp` 2D (temporarily used) padded rotated image, need this because rot{l90, 180, r90} require 2 different input args.
- `exp_mumapr` 2D exponential rotated mumap
"""
struct Workarray_s
    padimg::AbstractArray{<:Real, 2} # 2D padded image, (nx + padleft + padright, nz + padup + paddown)
    pad_imgr::AbstractArray{<:Real, 2} # 2D padded rotated image, (nx + 2 * pad_rotate_x, ny + 2 * pad_rotate_y)
    pad_imgr_tmp::AbstractArray{<:Real, 2} # 2D (temporarily used) padded rotated image, (nx + 2 * pad_rotate_x, ny + 2 * pad_rotate_y)
    exp_mumapr::AbstractArray{<:Real, 2} # 2D exp rotated mumap, (nx, nz)
    function Workarray_s(plan::SPECTplan)
            # allocate working buffers for each thread:
            # padimg is used in convolution with psfs
            padimg = zeros(promote_type(eltype(plan.mumap), Float32),
                            plan.nx + plan.padleft + plan.padright, plan.nz + plan.padup + plan.paddown)
            # pad_imgr stores 2D rotated & padded image
            pad_imgr = zeros(promote_type(eltype(plan.mumap), Float32),
                            plan.nx + 2 * plan.pad_rotate_x,
                            plan.ny + 2 * plan.pad_rotate_y)
            # pad_imgr_tmp stores 2D (temporarily used) rotated & padded image, need this in inplace rotation operation
            pad_imgr_tmp = zeros(promote_type(eltype(plan.mumap), Float32),
                            plan.nx + 2 * plan.pad_rotate_x,
                            plan.ny + 2 * plan.pad_rotate_y)
            # exp_mumapr stores 2D exponential mumap in different view angles
            exp_mumapr = zeros(promote_type(eltype(plan.mumap), Float32), plan.nx, plan.nz)

            new(padimg, pad_imgr, pad_imgr_tmp, exp_mumapr)

    end
end


"""
    my_conv!(img, ker, padimg, plan)
    Convolve an image with a kernel using plan
"""
function my_conv!(img, ker, padimg, plan)
    # filter the image with a kernel, using replicate padding
    imfilter!(padimg, plan.padrepl(img), ker, NoPad(), plan.alg)
    return @view padimg[1:plan.nx, 1:plan.nz]
end

"""
    my_conv_adj!(img, ker, padimg, plan)
    The adjoint of convolving an image with a kernel using plan
"""
function my_conv_adj!(img, ker, padimg, plan)
    # filter the image with a kernel, using zero padding
    imfilter!(padimg, plan.padzero(img), ker, NoPad(), plan.alg)
    # adjoint of replicate padding
    padimg[1:1, :] .+= sum(padimg[plan.nx + plan.padright + 1:end, :], dims = 1)
    padimg[plan.nx:plan.nx, :] .+= sum(padimg[plan.nx + 1:plan.nx + plan.padright, :], dims = 1)
    padimg[:, 1:1] .+= sum(padimg[:, plan.nz + plan.paddown + 1:end], dims = 2)
    padimg[:, plan.nz:plan.nz] .+= sum(padimg[:, plan.nz + 1:plan.nz + plan.paddown], dims = 2)
    return @view padimg[1:plan.nx, 1:plan.nz]
end

"""
    project!(view, plan, workarray, image, viewidx)
    project a single view
"""
function project!(
    view::AbstractMatrix{<:AbstractFloat},
    plan::SPECTplan,
    workarray::Vector{Workarray_s},
    image::AbstractArray{<:Real,3},
    viewidx::Int
)

    for zout = 1:plan.ncore_iter_z
        Threads.@threads for zin = 1:plan.ncore_array_z[zout]
            # rotate image
            idx = (zout - 1) * plan.ncore + zin
            thid = Threads.threadid()

            copyto!((@view plan.imgr[:, :, idx]),
                    plan.rotateforw!(workarray[thid].pad_imgr,
                                    workarray[thid].pad_imgr_tmp,
                                    (@view image[:, :, idx]),
                                    plan.viewangle[viewidx],
                                    plan.nx,
                                    plan.ny,
                                    plan.pad_rotate_x,
                                    plan.pad_rotate_y))

            # rotate mumap

            copyto!((@view plan.mumapr[:, :, idx]),
                    plan.rotateforw!(workarray[thid].pad_imgr,
                                    workarray[thid].pad_imgr_tmp,
                                    (@view plan.mumap[:, :, idx]),
                                    plan.viewangle[viewidx],
                                    plan.nx,
                                    plan.ny,
                                    plan.pad_rotate_x,
                                    plan.pad_rotate_y))
        end
    end


    for yout = 1:plan.ncore_iter_y
        Threads.@threads for yin = 1:plan.ncore_array_y[yout]

            # account for half of the final slice thickness
            idx = (yout - 1) * plan.ncore + yin
            thid = Threads.threadid()
            workarray[thid].exp_mumapr .= - plan.mumapr[:, idx, :] / 2

            for j = 1:idx
                workarray[thid].exp_mumapr .+= plan.mumapr[:, j, :]
            end
            workarray[thid].exp_mumapr .*= - plan.dy
            workarray[thid].exp_mumapr .= exp.(workarray[thid].exp_mumapr)

            # convolve img with psf and consider attenuation, then store the result in view
            view .+= my_conv!(broadcast!(*, (@view plan.imgr[:, idx, :]),
                                            (@view plan.imgr[:, idx, :]),
                                            workarray[thid].exp_mumapr),
                              (@view plan.psfs[:, :, idx, viewidx]),
                              workarray[thid].padimg,
                              plan)
        end
    end
    return view
end


"""
    project!(views, plan, workarray, image ; index)
    project multiple views
"""
function project!(
    views::AbstractArray{<:AbstractFloat,3},
    plan::SPECTplan,
    workarray::Vector{Workarray_s},
    image::AbstractArray{<:Real,3} ;
    index::AbstractVector{<:Int} = 1:plan.nview, # all views
)

    # loop over each view index
    for i in index
        project!((@view views[:,:,i]), plan, workarray, image, i)
    end
    return views
end

"""
    views = project(plan, workarray, image ; kwargs...)
    initialize views
"""
function project(
    plan::SPECTplan,
    workarray::Vector{Workarray_s},
    image::AbstractArray{<:Real,3} ;
    kwargs...,
)
    views = zeros(promote_type(eltype(image), Float32), plan.nx, plan.nz, plan.nview)
    return project!(views, plan, workarray, image ; kwargs...)
end


"""
    views = project(image, mumap, psfs, nview, dy, interpidx; kwargs...)
    initialize plan and workarray
"""
function project(
    image::AbstractArray{<:Real,3},
    mumap::AbstractArray{<:Real}, # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:Real},
    nview::Int,
    dy::Float32;
    interpidx::Int = 1,
    kwargs...,
)
    plan = SPECTplan(mumap, psfs, nview, dy; interpidx)
    workarray = Vector{Workarray_s}(undef, plan.ncore)
    for i = 1:plan.ncore
        workarray[i] = Workarray_s(plan) # allocate
    end
    project(plan, workarray, image ; kwargs...)
end

"""
    backproject!(view, plan, workarray, viewidx)
    backproject a single view
"""
function backproject!(
    view::AbstractMatrix{<:AbstractFloat},
    plan::SPECTplan,
    workarray::Vector{Workarray_s},
    viewidx::Int
)

    # rotate mumap
    for zout = 1:plan.ncore_iter_z
        Threads.@threads for zin = 1:plan.ncore_array_z[zout]
            # rotate mumap
            idx = (zout - 1) * plan.ncore + zin
            thid = Threads.threadid()
            copyto!((@view plan.mumapr[:, :, idx]),
                    plan.rotateforw!(workarray[thid].pad_imgr,
                                    workarray[thid].pad_imgr_tmp,
                                    (@view plan.mumap[:, :, idx]),
                                    plan.viewangle[viewidx],
                                    plan.nx,
                                    plan.ny,
                                    plan.pad_rotate_x,
                                    plan.pad_rotate_y))
        end
    end

    # adjoint of convolving img with psf and considering attenuation map
    for yout = 1:plan.ncore_iter_y
        Threads.@threads for yin = 1:plan.ncore_array_y[yout]

            # account for half of the final slice thickness
            idx = (yout - 1) * plan.ncore + yin
            thid = Threads.threadid()
            workarray[thid].exp_mumapr .= - plan.mumapr[:, idx, :] / 2

            for j = 1:idx
                workarray[thid].exp_mumapr .+= plan.mumapr[:, j, :]
            end
            workarray[thid].exp_mumapr .*= - plan.dy
            workarray[thid].exp_mumapr .= exp.(workarray[thid].exp_mumapr)

            broadcast!(*, (@view plan.imgr[:, idx, :]),
                            my_conv_adj!(view,
                                        (@view plan.psfs[:, :, idx, viewidx]),
                                        workarray[thid].padimg,
                                        plan),
                            workarray[thid].exp_mumapr)
        end
    end

    # adjoint of rotating image

    for zout = 1:plan.ncore_iter_z
        Threads.@threads for zin = 1:plan.ncore_array_z[zout]
            idx = (zout - 1) * plan.ncore + zin
            thid = Threads.threadid()

            copyto!((@view plan.imgr[:, :, idx]),
                    plan.rotateadjt!(workarray[thid].pad_imgr,
                                    workarray[thid].pad_imgr_tmp,
                                    (@view plan.imgr[:, :, idx]),
                                    plan.viewangle[viewidx],
                                    plan.nx,
                                    plan.ny,
                                    plan.pad_rotate_x,
                                    plan.pad_rotate_y))
        end
    end

    return plan.imgr
end


"""
    backproject!(views, plan, workarray, image ; index)
    backproject multiple views
"""
function backproject!(
    views::AbstractArray{<:AbstractFloat,3},
    plan::SPECTplan,
    workarray::Vector{Workarray_s},
    image::AbstractArray{<:Real,3} ;
    index::AbstractVector{<:Int} = 1:plan.nview, # all views
)

    # loop over each view index
    for i in index
        image .+= backproject!(views[:,:,i], plan, workarray, i)
    end
    return image
end

"""
    image = backproject(plan, workarray, views ; index)
    initialize the image
"""
function backproject(
    plan::SPECTplan,
    workarray::Vector{Workarray_s},
    views::AbstractArray{<:Real,3} ;
    kwargs...,
)
    image = zeros(promote_type(eltype(views), Float32), plan.nx, plan.ny, plan.nz)
    return backproject!(views, plan, workarray, image ; kwargs...)
end


"""
    image = backproject(views, mumap, psfs, nview, dy, interpidx; kwargs...)
    initialize plan and workarray
"""
function backproject(
    views::AbstractArray{<:Real,3},
    mumap::AbstractArray{<:Real}, # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:Real},
    nview::Int,
    dy::Float32;
    interpidx::Int = 1,
    kwargs...,
)
    plan = SPECTplan(mumap, psfs, nview, dy; interpidx)
    workarray = Vector{Workarray_s}(undef, plan.ncore)
    for i = 1:plan.ncore
        workarray[i] = Workarray_s(plan) # allocate
    end
    backproject(plan, workarray, views; kwargs...)
end
