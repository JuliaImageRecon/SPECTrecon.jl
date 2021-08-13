# project.jl

"""
    SPECTplan
Struct for storing key factors for a SPECT system model
- `mumap [nx,ny,nz]` attenuation map, must be 3D, possibly zeros()
- `psfs [nx_psf,nz_psf,ny,nview]` point spread function, must be 4D, with `nx_psf` and `nz_psf` odd, and symmetric for each slice
- `nview` number of views, must be integer
- `ncore` number of cores, must be integer
- `viewangle` set of view angles, must be from 0 to 2π
- `interpidx` interpolation method, 1 is 1d, 2 is 2d
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
- `imgr` 3D rotated image
- `mumapr` 3D rotated mumap
- `T` datatype of work arrays
- `fft_plan` plan for doing fft, see plan_fft!
- `ifft_plan` plan for doing ifft, see plan_ifft!
Currently code assumes each of the `nview` projection views is `[nx,nz]`
Currently code assumes `nx = ny`
Currently code assumes uniform angular sampling
Currently imrotate3 code only supports linear interpolation
Currently code uses multiprocessing using # of cores specified by Threads.nthreads() in Julia
Currently code calls imfilter! function which is not a fully in-place operation, so it will allocate some extra memories
"""
struct SPECTplan
    mumap::AbstractArray{<:Real, 3} # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:Real, 4} # PSFs must be 4D, [nx_psf, nz_psf, ny, nview], finally be centered psf
    nview::Int # number of views
    ncore::Int # number of cores
    viewangle::AbstractVector
    interpidx::Int
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
    imgr::AbstractArray{<:Real, 3} # 3D rotated image, (nx, ny, nz)
    mumapr::AbstractArray{<:Real, 3} # 3D rotated mumap, (nx, ny, nz)
    T::DataType # default type for work arrays etc.
    fft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}
    ifft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}
    # other options for how to do the projection?
    function SPECTplan(mumap::AbstractArray{<:Real,3},
                        psfs::AbstractArray{<:Real,4},
                        nview::Int,
                        dy::RealU ;
                        viewangle::AbstractVector = (0:nview - 1) / nview * (2π), # set of view angles
                        interpidx::Int = 2, # 1 is for 1d interpolation, 2 is for 2d interpolation
                        padleft::Int = _padleft(mumap, psfs),
                        padright::Int = _padright(mumap, psfs),
                        padup::Int = _padup(mumap, psfs),
                        paddown::Int = _paddown(mumap, psfs),
                        T::DataType = promote_type(eltype(mumap), Float32)
                        )
        # check nx = ny ? typically 128 x 128 x 81
        nx, ny, nz = size(mumap)
        nx_psf = size(psfs, 1)
        nz_psf = size(psfs, 2)
        @assert isequal(nx, ny)
        @assert iseven(nx) && iseven(ny)
        @assert isodd(nx_psf) && isodd(nz_psf)
        @assert all(mapslices(x -> x == reverse(x), psfs, dims = [1, 2]))
        # center the psfs
        psfs = OffsetArray(psfs,
                        OffsetArrays.Origin(-Int((nx_psf-1)/2),
                                            -Int((nz_psf-1)/2),
                                            1,
                                            1)
                           )
        @assert interpidx == 1 || interpidx == 2
        pad_rotate_x = ceil(Int, 1 + nx * sqrt(2)/2 - nx / 2)
        pad_rotate_y = ceil(Int, 1 + ny * sqrt(2)/2 - ny / 2)
        # mypad1 = x -> padarray(x, Pad(:replicate, (nx_psf, nz_psf), (nx_psf, nz_psf)))
        padrepl = x -> OffsetArrays.no_offset_view(BorderArray(x, Pad(:replicate, (padleft, padup), (padright, paddown))))
        padzero = x -> OffsetArrays.no_offset_view(BorderArray(x, Fill(0, (padleft, padup), (padright, paddown))))

        ncore = Threads.nthreads()

        # allocate working buffers:
        # imgr stores 3D image in different view angles
        imgr = zeros(T, nx, ny, nz)
        # mumapr stores 3D mumap in different view angles
        mumapr = similar(imgr)
        tmp = Array{Complex{T}}(undef, nx + padleft + padright, nz + padup + paddown)
        fft_plan = plan_fft!(tmp)
        ifft_plan = plan_ifft!(tmp)

        new(mumap, psfs, nview, ncore, viewangle, interpidx, dy,
            nx, ny, nz, nx_psf, nz_psf, padrepl, padzero, padleft, padright,
            padup, paddown, pad_rotate_x, pad_rotate_y, imgr, mumapr, T,
            fft_plan, ifft_plan)
        #  creates objects of the block's type (inner constructor methods).
    end
end

"""
    Workarray_s
Struct for storing keys of the work array for a single thread
add tmp vectors to avoid allocating in rotate_x and rotate_y
- `padimg` 2D padded image for imfilter3
- `img_compl` 2D complex image for fft
- `ker_compl` 2D complex image for fft
- `pad_imgr` padded 2D rotated image
- `pad_imgr_tmp` 2D (temporarily used) padded rotated image, need this because some functions require 2 different input args.
- `exp_mumapr` 2D exponential rotated mumap
- `vec_rotate_x` 1D vector storing rotated axis in rotate_x function
- `vec_rotate_y` 1D vector storing rotated axis in rotate_y function
- `interp_x` sparse interpolator for rotating in x direction
- `interp_y` sparse interpolator for rotating in y direction
"""
struct Workarray_s
    padimg::AbstractArray{<:Real, 2} # 2D padded image, (nx + padleft + padright, nz + padup + paddown)
    img_compl::AbstractArray{<:ComplexF32, 2} # 2D complex image, the same size as padimg
    ker_compl::AbstractArray{<:ComplexF32, 2} # 2D complex image, the same size as padimg
    pad_imgr::AbstractArray{<:Real, 2} # 2D padded rotated image, (nx + 2 * pad_rotate_x, ny + 2 * pad_rotate_y)
    pad_imgr_tmp::AbstractArray{<:Real, 2} # 2D (temporarily used) padded rotated image, (nx + 2 * pad_rotate_x, ny + 2 * pad_rotate_y)
    exp_mumapr::AbstractArray{<:Real, 2} # 2D exp rotated mumap, (nx, nz)
    vec_rotate_x::Union{Nothing, AbstractVector}
    vec_rotate_y::Union{Nothing, AbstractVector}
    interp_x::Union{Nothing, SparseInterpolator}
    interp_y::Union{Nothing, SparseInterpolator}
    function Workarray_s(plan::SPECTplan)
            # allocate working buffers for each thread:
            # padimg is used in convolution with psfs
            padimg = zeros(plan.T,
                            plan.nx + plan.padleft + plan.padright,
                            plan.nz + plan.padup + plan.paddown)
            # complex padimg
            img_compl = similar(padimg, Complex{plan.T})
            # complex kernel
            ker_compl = similar(img_compl)
            # pad_imgr stores 2D rotated & padded image
            pad_imgr = zeros(plan.T,
                            plan.nx + 2 * plan.pad_rotate_x,
                            plan.ny + 2 * plan.pad_rotate_y)
            # pad_imgr_tmp stores 2D (temporarily used) rotated & padded image, need this in inplace rotation operation
            pad_imgr_tmp = similar(pad_imgr)
            # exp_mumapr stores 2D exponential mumap in different view angles
            exp_mumapr = zeros(plan.T, plan.nx, plan.nz)
            if plan.interpidx == 1
                # vec_rotate_x and vec_rotate_y store rotated axis
                vec_rotate_x = zeros(plan.T, plan.nx + 2 * plan.pad_rotate_x)
                vec_rotate_y = zeros(plan.T, plan.ny + 2 * plan.pad_rotate_y)

                interp_x = SparseInterpolator(LinearSpline(plan.T), vec_rotate_x, length(vec_rotate_x))
                interp_y = SparseInterpolator(LinearSpline(plan.T), vec_rotate_y, length(vec_rotate_y))
                new(padimg, img_compl, ker_compl,
                    pad_imgr, pad_imgr_tmp, exp_mumapr,
                    vec_rotate_x, vec_rotate_y, interp_x, interp_y)
            else
                new(padimg, img_compl, ker_compl,
                    pad_imgr, pad_imgr_tmp, exp_mumapr,
                    nothing, nothing, nothing, nothing)
            end
    end
end

"""
    my_conv!(padimg, img, ker, img_compl, ker_compl, plan)
Convolve `img` with `ker` and store in `padimg`
`img_compl`, `ker_compl`, `tmp_compl` are used in fft operations
"""
function my_conv!(padimg::AbstractArray{<:Real, 2},
                  img::AbstractArray{<:Real, 2},
                  ker::AbstractArray{<:Real, 2},
                  img_compl::AbstractArray{<:ComplexF32, 2},
                  ker_compl::AbstractArray{<:ComplexF32, 2},
                  plan::SPECTplan)
    # filter the image with a kernel, using replicate padding and fft convolution
    padimg .= plan.padrepl(img)
    fft_plan = plan_fft!(img_compl)
    ifft_plan = plan_ifft!(img_compl)
    imfilter3!(padimg, ker, img_compl, ker_compl, fft_plan, ifft_plan)
    return @view padimg[1+plan.padleft:plan.nx+plan.padleft,
                        1+plan.padup:plan.nz+plan.padup]
end

"""
    my_conv_adj!(padimg, img, ker, img_compl, ker_compl, plan)
The adjoint of convolving `img` with `ker` and storing in `padimg`
`img_compl`, `ker_compl`, `tmp_compl` are used in fft operations
"""
function my_conv_adj!(padimg::AbstractArray{<:Real, 2},
                  img::AbstractArray{<:Real, 2},
                  ker::AbstractArray{<:Real, 2},
                  img_compl::AbstractArray{<:ComplexF32, 2},
                  ker_compl::AbstractArray{<:ComplexF32, 2},
                  plan::SPECTplan)
    # filter the image with a kernel, using zero padding and fft convolution
    padimg .= plan.padzero(img)
    imfilter3!(padimg, ker, img_compl, ker_compl, plan.fft_plan, plan.ifft_plan)
    # adjoint of replicate padding
    broadcast!(+, (@view padimg[1+plan.padleft:1+plan.padleft, :]),
                  (@view padimg[1+plan.padleft:1+plan.padleft, :]),
                   sum((@view padimg[1:plan.padleft, :]), dims = 1))
    broadcast!(+, (@view padimg[plan.nx+plan.padleft:plan.nx+plan.padleft, :]),
                  (@view padimg[plan.nx+plan.padleft:plan.nx+plan.padleft, :]),
                  sum((@view padimg[plan.nx+plan.padleft+1:end, :]), dims = 1))
    broadcast!(+, (@view padimg[:, 1+plan.padup:1+plan.padup]),
                  (@view padimg[:, 1+plan.padup:1+plan.padup]),
                  sum((@view padimg[:, 1:plan.padup]), dims = 2))
    broadcast!(+, (@view padimg[:, plan.nz+plan.padup:plan.nz+plan.padup]),
                  (@view padimg[:, plan.nz+plan.padup:plan.nz+plan.padup]),
                  sum((@view padimg[:, plan.nz+plan.padup+1:end]), dims = 2))
    return @view padimg[1+plan.padleft:plan.nx+plan.padleft,
                        1+plan.padup:plan.nz+plan.padup]
end

"""
    project!(view, plan, workarray, image, viewidx)
Project a single view
"""
function project!(
    view::AbstractMatrix{<:AbstractFloat},
    plan::SPECTplan,
    workarray::Vector{Workarray_s},
    image::AbstractArray{<:Real,3},
    viewidx::Int
)
    # rotate image and mumap using multiple processors
    Threads.@threads for z = 1:plan.nz

        thid = Threads.threadid()
        if plan.interpidx == 1
            # rotate image and store in plan.imgr
            (@view plan.imgr[:, :, z]) .= imrotate3!(workarray[thid].pad_imgr,
                                                     workarray[thid].pad_imgr_tmp,
                                                     (@view image[:, :, z]),
                                                     plan.viewangle[viewidx],
                                                     workarray[thid].interp_x,
                                                     workarray[thid].interp_y,
                                                     workarray[thid].vec_rotate_x,
                                                     workarray[thid].vec_rotate_y)

        # rotate mumap and store in plan.mumapr

            (@view plan.mumapr[:, :, z]) .= imrotate3!(workarray[thid].pad_imgr,
                                                       workarray[thid].pad_imgr_tmp,
                                                       (@view plan.mumap[:, :, z]),
                                                       plan.viewangle[viewidx],
                                                       workarray[thid].interp_x,
                                                       workarray[thid].interp_y,
                                                       workarray[thid].vec_rotate_x,
                                                       workarray[thid].vec_rotate_y)
        else
            # rotate image and store in plan.imgr

            (@view plan.imgr[:, :, z]) .= imrotate3!(workarray[thid].pad_imgr,
                                                     workarray[thid].pad_imgr_tmp,
                                                     (@view image[:, :, z]),
                                                     plan.viewangle[viewidx])

        # rotate mumap and store in plan.mumapr

            (@view plan.mumapr[:, :, z]) .= imrotate3!(workarray[thid].pad_imgr,
                                                       workarray[thid].pad_imgr_tmp,
                                                       (@view plan.mumap[:, :, z]),
                                                       plan.viewangle[viewidx])

        end
    end

    Threads.@threads for y = 1:plan.ny
        thid = Threads.threadid()
        # account for half of the final slice thickness
        workarray[thid].exp_mumapr .= (-0.5) .* (@view plan.mumapr[:, y, :])
        broadcast!(+, workarray[thid].exp_mumapr,
                      workarray[thid].exp_mumapr,
                      dropdims(sum((@view plan.mumapr[:, 1:y, :]), dims = 2); dims = 2))
        broadcast!(*, workarray[thid].exp_mumapr, workarray[thid].exp_mumapr, - plan.dy)
        workarray[thid].exp_mumapr .= exp.(workarray[thid].exp_mumapr)
        # apply depth-dependent attenuation
        broadcast!(*, (@view plan.imgr[:, y, :]), (@view plan.imgr[:, y, :]), workarray[thid].exp_mumapr)

        # convolve img with psf and add up to view
        broadcast!(+, view, view, my_conv!(workarray[thid].padimg,
                                           (@view plan.imgr[:, y, :]),
                                           (@view plan.psfs[:, :, y, viewidx]),
                                           workarray[thid].img_compl,
                                           workarray[thid].ker_compl,
                                           plan))
    end
    return view
end


"""
    project!(views, plan, workarray, image ; index)
Project multiple views
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
Initialize views
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
    views = project(image, mumap, psfs, nview, dy; interpidx, kwargs...)
Initialize plan and workarray
"""
function project(
    image::AbstractArray{<:Real,3},
    mumap::AbstractArray{<:Real}, # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:Real},
    nview::Int,
    dy::Float32;
    interpidx::Int = 2,
    kwargs...,
)
    plan = SPECTplan(mumap, psfs, nview, dy; interpidx, kwargs...)
    workarray = Vector{Workarray_s}(undef, plan.ncore)
    for i = 1:plan.ncore
        workarray[i] = Workarray_s(plan) # allocate
    end
    project(plan, workarray, image ; kwargs...)
end

"""
    backproject!(view, plan, workarray, viewidx)
Backproject a single view
"""
function backproject!(
    view::AbstractMatrix{<:AbstractFloat},
    plan::SPECTplan,
    workarray::Vector{Workarray_s},
    viewidx::Int
)

    # rotate mumap and store in plan.mumapr
    Threads.@threads for z = 1:plan.nz
        # get thread id
        thid = Threads.threadid()
        if plan.interpidx == 1

            (@view plan.mumapr[:, :, z]) .= imrotate3!(workarray[thid].pad_imgr,
                                                       workarray[thid].pad_imgr_tmp,
                                                       (@view plan.mumap[:, :, z]),
                                                       plan.viewangle[viewidx],
                                                       workarray[thid].interp_x,
                                                       workarray[thid].interp_y,
                                                       workarray[thid].vec_rotate_x,
                                                       workarray[thid].vec_rotate_y)
        else

            (@view plan.mumapr[:, :, z]) .= imrotate3!(workarray[thid].pad_imgr,
                                                       workarray[thid].pad_imgr_tmp,
                                                       (@view plan.mumap[:, :, z]),
                                                       plan.viewangle[viewidx])
        end

    end

    # adjoint of convolving img with psf and applying attenuation map
    Threads.@threads for y = 1:plan.ny
        # get thread id
        thid = Threads.threadid()
        # account for half of the final slice thickness
        workarray[thid].exp_mumapr .= (@view plan.mumapr[:, y, :])
        broadcast!(*, workarray[thid].exp_mumapr, workarray[thid].exp_mumapr, -0.5)
        broadcast!(+, workarray[thid].exp_mumapr,
                      workarray[thid].exp_mumapr,
                      dropdims(sum((@view plan.mumapr[:, 1:y, :]), dims = 2); dims = 2))
        broadcast!(*, workarray[thid].exp_mumapr, workarray[thid].exp_mumapr, - plan.dy)
        workarray[thid].exp_mumapr .= exp.(workarray[thid].exp_mumapr)


        (@view plan.imgr[:, y, :]) .= my_conv_adj!(workarray[thid].padimg,
                                                   view,
                                                   (@view plan.psfs[:, :, y, viewidx]),
                                                   workarray[thid].img_compl,
                                                   workarray[thid].ker_compl,
                                                   plan)
        broadcast!(*, (@view plan.imgr[:, y, :]), (@view plan.imgr[:, y, :]), workarray[thid].exp_mumapr)


    end

    # adjoint of rotating image
    Threads.@threads for z = 1:plan.nz
        # get thread id
        thid = Threads.threadid()
        if plan.interpidx == 1
            (@view plan.imgr[:, :, z]) .= imrotate3_adj!(workarray[thid].pad_imgr,
                                                         workarray[thid].pad_imgr_tmp,
                                                         (@view plan.imgr[:, :, z]),
                                                         plan.viewangle[viewidx],
                                                         workarray[thid].interp_x,
                                                         workarray[thid].interp_y,
                                                         workarray[thid].vec_rotate_x,
                                                         workarray[thid].vec_rotate_y)
        else
            (@view plan.imgr[:, :, z]) .= imrotate3_adj!(workarray[thid].pad_imgr,
                                                         workarray[thid].pad_imgr_tmp,
                                                         (@view plan.imgr[:, :, z]),
                                                         plan.viewangle[viewidx])
        end
    end

    return plan.imgr
end


"""
    backproject!(views, plan, workarray, image ; index)
Backproject multiple views
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
        broadcast!(+, image, image, backproject!((@view views[:,:,i]), plan, workarray, i))
    end
    return image
end

"""
    image = backproject(plan, workarray, views ; kwargs...)
Initialize the image
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
    image = backproject(views, mumap, psfs, nview, dy; interpidx, kwargs...)
Initialize plan and workarray
"""
function backproject(
    views::AbstractArray{<:Real,3},
    mumap::AbstractArray{<:Real}, # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:Real},
    nview::Int,
    dy::Float32;
    interpidx::Int = 2,
    kwargs...,
)
    plan = SPECTplan(mumap, psfs, nview, dy; interpidx, kwargs...)
    workarray = Vector{Workarray_s}(undef, plan.ncore)
    for i = 1:plan.ncore
        workarray[i] = Workarray_s(plan) # allocate
    end
    backproject(plan, workarray, views; kwargs...)
end
