# project.jl

include("helper.jl")
include("rotate3.jl")

const RealU = Number # Union{Real, Unitful.Length}

"""
    SPECTplan
Struct for storing key factors for a SPECT system model
- `mumap [nx,ny,nz]` attenuation map, must be 3D, possibly zeros()
- `psfs [nx_psf,nz_psf,ny,nview]` point spread function, must be 4D, with `nx_psf` and `nz_psf` odd, and symmetric for each slice
- `nview` number of views, must be integer
- `rotateforw!` rotation method, default is using 1d linear interpolation
- `rotateadjt!` adjoint of rotation method, default is using 1d linear interpolation
- `viewangle` set of view angles, must be from 0 to 2π
- `dy` voxel size in y direction (dx is the same value)
- `nx` number of voxels in x direction of the image, must be integer
- `ny` number of voxels in y direction of the image, must be integer
- `nz` number of voxels in z direction of the image, must be integer
- `nx_psf` number of voxels in x direction of the psf, must be integer
- `nz_psf` number of voxels in z direction of the psf, must be integer
- `padrepl` padding function using replicate border condition
- `padzero` padding function using zero border condition
- {padleft,padright,padup,paddown} pixels padded along {left,right,up,down} direction for convolution with psfs
- `alg` algorithms used for convolution, default is FFT
- `padimg` 2D padded image for imfilter
- `imgr` 3D rotated image
- `pad_imgr` padded 2D rotated image
- `pad_imgr_tmp` 2D tmp padded rotated image, need this because rot{l90, 180, r90} require 2 different input args.
- `mumapr` 3D rotated mumap
- `exp_mumapr` 2D exponential rotated mumap
Currently code assumes each of the `nview` projection views is `[nx,nz]`
Currently code assumes `nx = ny`
Currently code assumes uniform angular sampling
Currently imrotate3 code only supports linear interpolation
"""
struct SPECTplan
    mumap::AbstractArray{<:Real, 3} # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:Real, 4} # PSFs must be 4D, [nx_psf, nz_psf, ny, nview], finally be centered psf
    nview::Int
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
    padimg::AbstractArray{<:Real, 2} # 2D padded image, (nx + padleft + padright, nz + padup + paddown)
    imgr::AbstractArray{<:Real, 3} # 3D rotated image, (nx, ny, nz)
    pad_imgr::AbstractArray{<:Real, 2} # 2D padded rotated image, (nx + 2 * pad_rotate_x, ny + 2 * pad_rotate_y)
    pad_imgr_tmp::AbstractArray{<:Real, 2} # 2D tmp padded rotated image, (nx + 2 * pad_rotate_x, ny + 2 * pad_rotate_y)
    mumapr::AbstractArray{<:Real, 3} # 3D rotated mumap, (nx, ny, nz)
    exp_mumapr::AbstractArray{<:Real, 2} # 2D exp rotated mumap, (nx, nz)
    # other options for how to do the projection?
    function SPECTplan(mumap::AbstractArray{<:Real,3},
                        psfs::AbstractArray{<:Real,4},
                        nview::Int,
                        dy::RealU;
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


        # allocate working buffers:
        # padimg is used in convolution with psfs
        padimg = zeros(promote_type(eltype(mumap), Float32), nx + padleft + padright, nz + padup + paddown)
        # imgr stores 3D image in different view angles
        imgr = zeros(promote_type(eltype(mumap), Float32), nx, ny, nz)
        # pad_imgr stores 2D rotated & padded image
        pad_imgr = zeros(promote_type(eltype(mumap), Float32),
                        nx + 2 * pad_rotate_x,
                        ny + 2 * pad_rotate_y)
        # pad_imgr_tmp stores 2D rotated & padded image, used in inplace rotation operation
        pad_imgr_tmp = zeros(promote_type(eltype(mumap), Float32),
                        nx + 2 * pad_rotate_x,
                        ny + 2 * pad_rotate_y)
        # mumapr stores 3D mumap in different view angles
        mumapr = zeros(promote_type(eltype(mumap), Float32), nx, ny, nz)
        # exp_mumapr stores 2D exponential mumap in different view angles
        exp_mumapr = zeros(promote_type(eltype(mumap), Float32), nx, nz)

        if conv_alg === :fft
            alg = Algorithm.FFT()
        elseif conv_alg === :fir
            alg = Algorithm.FIR()
        else
            throw("invalid convolution algorithm choice!")
        end
        new(mumap, psfs, nview, rotateforw!, rotateadjt!, viewangle, dy, nx, ny, nz,
                nx_psf, nz_psf, padrepl, padzero, padleft, padright, padup, paddown,
                pad_rotate_x, pad_rotate_y, alg, padimg, imgr, pad_imgr, pad_imgr_tmp,
                mumapr, exp_mumapr)
        #  creates objects of the block's type (inner constructor methods).
    end
end


"""
    my_conv!(plan, img, ker, i, viewidx)
    Convolve an image with a kernel using plan
"""
function my_conv!(plan, img, ker, i, viewidx)
    imfilter!(plan.padimg, plan.padrepl(img), plan.psfs[:, :, i, viewidx], NoPad(), plan.alg)
    return @view plan.padimg[1:plan.nx, 1:plan.nz]
end

"""
    my_conv_adj!(plan, img, ker, i, viewidx)
    The adjoint of convolving an image with a kernel using plan
"""
function my_conv_adj!(plan, img, ker, i, viewidx)
    imfilter!(plan.padimg, plan.padzero(img), plan.psfs[:, :, i, viewidx], NoPad(), plan.alg)
    plan.padimg[1:1, :] .+= sum(plan.padimg[plan.nx + plan.padright + 1:end, :], dims = 1)
    plan.padimg[plan.nx:plan.nx, :] .+= sum(plan.padimg[plan.nx + 1:plan.nx + plan.padright, :], dims = 1)
    plan.padimg[:, 1:1] .+= sum(plan.padimg[:, plan.nz + plan.paddown + 1:end], dims = 2)
    plan.padimg[:, plan.nz:plan.nz] .+= sum(plan.padimg[:, plan.nz + 1:plan.nz + plan.paddown], dims = 2)
    return @view plan.padimg[1:plan.nx, 1:plan.nz]
end

"""
    project!(view, plan, image, viewidx)
    1. Project a single view
"""
function project!(
    view::AbstractMatrix{<:AbstractFloat},
    plan::SPECTplan,
    image::AbstractArray{<:Real,3},
    viewidx::Int,
)

    for z = 1:plan.nz
        # rotate image
        plan.imgr[:, :, z] .= plan.rotateforw!(plan.pad_imgr, plan.pad_imgr_tmp,
                                    image[:,:,z], plan.viewangle[viewidx], plan.nx,
                                    plan.ny, plan.pad_rotate_x, plan.pad_rotate_y)
        # rotate mumap
        plan.mumapr[:, :, z] .= plan.rotateforw!(plan.pad_imgr, plan.pad_imgr_tmp,
                                    plan.mumap[:,:,z], plan.viewangle[viewidx], plan.nx,
                                    plan.ny, plan.pad_rotate_x, plan.pad_rotate_y)
    end

    for i = 1:plan.ny
        # account for half of the final slice thickness
        plan.exp_mumapr .= - plan.mumapr[:, i, :] / 2
        for j = 1:i
            plan.exp_mumapr .+= plan.mumapr[:, j, :]
        end
        plan.exp_mumapr .*= - plan.dy
        plan.exp_mumapr .= exp.(plan.exp_mumapr)

        view .+= my_conv!(plan, plan.imgr[:, i, :] .* plan.exp_mumapr, plan.psfs[:, :, i, viewidx], i, viewidx)
    end
    return view
end


"""
    project!(views, plan, image ; index)
    2. Project multiple views, call 1
"""
function project!(
    views::AbstractArray{<:AbstractFloat,3},
    plan::SPECTplan,
    image::AbstractArray{<:Real,3} ;
    index::AbstractVector{<:Int} = 1:plan.nview, # all views
)

    # loop over each view index
    for i in index
        project!((@view views[:,:,i]), plan, image, i)
    end
    return views
end

"""
    views = project(plan, image ; index)
    3. Initialize views, call 2
"""
function project(
    plan::SPECTplan,
    image::AbstractArray{<:Real,3} ;
    kwargs...,
)
    views = zeros(promote_type(eltype(image), Float32), plan.nx, plan.nz, plan.nview)
    return project!(views, plan, image ; kwargs...)
end


"""
    views = project(image, mumap, psfs, nview, interpidx; kwargs...), test the function, call 3
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
    project(plan, image; kwargs...)
end

"""
    backproject!(view, plan, viewidx)
    1. Backproject a single view
"""
function backproject!(
    view::AbstractMatrix{<:AbstractFloat},
    plan::SPECTplan,
    viewidx::Int
)

    for z = 1:plan.nz
        # rotate mumap
        plan.mumapr[:,:,z] .= plan.rotateforw!(plan.pad_imgr, plan.pad_imgr_tmp,
                                plan.mumap[:,:,z], plan.viewangle[viewidx], plan.nx,
                                plan.ny, plan.pad_rotate_x, plan.pad_rotate_y)
    end


    for i = 1:plan.ny

        # account for half of the final slice thickness
        plan.exp_mumapr .= - plan.mumapr[:, i, :] / 2
        for j = 1:i
            plan.exp_mumapr .+= plan.mumapr[:, j, :]
        end
        plan.exp_mumapr .*= - plan.dy
        plan.exp_mumapr .= exp.(plan.exp_mumapr)

        # adjoint of convolution, convolve with reverse of psfs
        # adjoint of multiplying with mumap
        plan.imgr[:, i, :] .= my_conv_adj!(plan, view, plan.psfs[:, :, i, viewidx], i, viewidx) .* plan.exp_mumapr
    end

    # adjoint of imrotate
    for z = 1:plan.nz
        plan.imgr[:,:,z] .= plan.rotateadjt!(plan.pad_imgr, plan.pad_imgr_tmp,
                                plan.imgr[:,:,z], plan.viewangle[viewidx], plan.nx,
                                plan.ny, plan.pad_rotate_x, plan.pad_rotate_y)
    end
    return plan.imgr
end


"""
    backproject!(views, plan, image ; index)
    2. Backproject multiple views, call 1
"""
function backproject!(
    views::AbstractArray{<:AbstractFloat,3},
    plan::SPECTplan,
    image::AbstractArray{<:Real,3} ;
    index::AbstractVector{<:Int} = 1:plan.nview, # all views
)

    # loop over each view index
    for i in index
        image .+= backproject!(views[:,:,i], plan, i)
    end
    return image
end

"""
    image = backproject(plan, views ; index)
    3. Initialize image, call 2
"""
function backproject(
    plan::SPECTplan,
    views::AbstractArray{<:Real,3} ;
    kwargs...,
)
    image = zeros(promote_type(eltype(views), Float32), plan.nx, plan.ny, plan.nz)
    return backproject!(views, plan, image ; kwargs...)
end


"""
    image = backproject(views, mumap, psfs, nview, interpidx; kwargs...), test the function, call 3
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
    backproject(plan, views; kwargs...)
end
