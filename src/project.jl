# project.jl
# Use BorderArray instead of PadArray
include("rotate3.jl")

"""
    SPECTplan
Struct for storing key factors for a SPECT system model
- `mumap [nx,ny,nz]` attenuation map, must be 3D, possibly zeros()
- `psfs [nx_psf,nz_psf,ny,nview]` point spread function, must be 4D, with `nx_psf` and `nz_psf` odd, and symmetric for each slice
- `nview` number of views, must be integer
- `rotateforw` rotation method, default is using 1d linear interpolation
- `rotateadjt` adjoint of rotation method, default is using 1d linear interpolation
- `dy` voxel size in y direction (dx is the same value)
- `nx` number of voxels in x direction of the image, must be integer
- `ny` number of voxels in y direction of the image, must be integer
- `nz` number of voxels in z direction of the image, must be integer
- `nx_psf` number of voxels in x direction of the psf, must be integer
- `nz_psf` number of voxels in z direction of the psf, must be integer
- `padrepl` padding function using replicate border condition
- `padzero` padding function using zero border condition
- {padleft,padright,padup,paddown} pixels padded along {left,right,up,down} direction
- `alg` algorithms used for convolution, default is FFT
- `workimg` 2D padded image for imfilter
- `imgr` 3D rotated image
- `mumapr` 3D rotated mumap
Currently code assumes each of the `nview` projection views is `[nx,nz]`
Currently code assumes `nx = ny`
Currently code assumes uniform angular sampling
Currently imrotate3 code only supports linear interpolation
"""
struct SPECTplan
    mumap::AbstractArray{<:Real, 3} # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:Real, 4} # PSFs must be 4D, [nx_psf, nz_psf, ny, nview], finally be centered psf
    nview::Int
    rotateforw::Function
    rotateadjt::Function
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
    workimg::AbstractArray{<:Real, 2} # padded image
    imgr::AbstractArray{<:Real, 3} # rotated image
    pad_imgr::AbstractArray{<:Real, 2} # 2D padded rotated image
    mumapr::AbstractArray{<:Real, 3} # rotated mumap
    # other options for how to do the projection?
    function SPECTplan(mumap,
                        psfs,
                        nview,
                        dy;
                        interpidx::Int = 1,
                        conv_alg::Symbol = :fft,
                        padleft::Int = 0,
                        padright::Int = 0,
                        padup::Int = 0,
                        paddown::Int = 0)
        # check nx = ny ? typically 128 x 128 x 81
        nx, ny, nz = size(mumap)
        nx_psf = size(psfs, 1)
        nz_psf = size(psfs, 2)
        @assert isequal(nx, ny)
        @assert iseven(nx) && iseven(ny)
        @assert isodd(nx_psf) && isodd(nz_psf)
        issym = x -> x == reverse(x)
        @assert all(mapslices(issym, psfs, dims = [1, 2]))
        # center the psfs
        psfs = OffsetArray(psfs, OffsetArrays.Origin(-Int((nx_psf-1)/2), -Int((nz_psf-1)/2), 1, 1))
        if interpidx == 1
            rotateforw = imrotate3!
            rotateadjt = imrotate3_adj!
        elseif interpidx == 2
            rotateforw = imrotate3emmt!
            rotateadjt = imrotate3emmt_adj!
        else
            throw("invalid interpidx!")
        end
        viewangle = (0:nview-1) / nview * (2π)

        Power2 = x -> 2^(ceil(Int, log2(x)))

        padleft = iszero(padleft) ? ceil(Int, (Power2(nx+nx_psf-1) - nx) / 2) : padleft
        padright = iszero(padright) ? floor(Int, (Power2(nx+nx_psf-1) - nx) / 2) : padright
        padup = iszero(padup) ? ceil(Int, (Power2(nz+nz_psf-1) - nz) / 2) : padup
        paddown = iszero(paddown) ? floor(Int, (Power2(nz+nz_psf-1) - nz) / 2) : paddown
        pad_rotate_x = ceil(Int, 1 + nx * sqrt(2)/2 - nx / 2)
        pad_rotate_y = ceil(Int, 1 + ny * sqrt(2)/2 - ny / 2)
        # mypad1 = x -> padarray(x, Pad(:replicate, (nx_psf, nz_psf), (nx_psf, nz_psf)))
        padrepl = x -> BorderArray(x, Pad(:replicate, (padleft, padup), (padright, paddown)))
        padzero = x -> BorderArray(x, Fill(0, (padleft, padup), (padright, paddown)))

        # mypad = x -> mypad2(mypad1(x))
        workimg = zeros(promote_type(eltype(mumap), Float32), nx + padleft + padright, nz + padup + paddown)
        imgr = zeros(promote_type(eltype(mumap), Float32), nx, ny, nz)
        pad_imgr = zeros(promote_type(eltype(mumap), Float32),
                        nx + 2 * pad_rotate_x,
                        ny + 2 * pad_rotate_y)
        mumapr = zeros(promote_type(eltype(mumap), Float32), nx, ny, nz)

        if conv_alg === :fft
            alg = Algorithm.FFT()
        elseif conv_alg === :fir
            alg = Algorithm.FIR()
        else
            throw("unknown convolution algorithm choice!")
        end
        new(mumap, psfs, nview, rotateforw, rotateadjt, viewangle, dy, nx, ny, nz,
                nx_psf, nz_psf, padrepl, padzero, padleft, padright, padup, paddown,
                pad_rotate_x, pad_rotate_y, alg, workimg, imgr, pad_imgr, mumapr)
        #  creates objects of the block's type (inner constructor methods).
    end
end

"""
    my_conv(img, ker, plan)
    Convolve an image with a kernel
"""
function my_conv(img, ker, plan, i, viewidx)
    return imfilter!(plan.workimg, plan.padrepl(img),
                    plan.psfs[:, :, i, viewidx], NoPad(), plan.alg)[1:plan.nx, 1:plan.nz]
    # return imfilter(plan.padrepl(img), plan.psfs[:, :, i, viewidx], plan.alg)[1:plan.nx, 1:plan.nz]
end
"""
    my_conv_adj(img, ker, plan)
"""
function my_conv_adj(img, ker, plan, i, viewidx)
    # tmp = imfilter(plan.padzero(img), centered(ker), plan.alg)
    # tmp[1:1, :] .+= sum(tmp[1 - plan.padleft:0, :], dims = 1)
    # tmp[plan.nx:plan.nx, :] .+= sum(tmp[plan.nx + 1:end, :], dims = 1)
    # tmp[:, 1:1] .+= sum(tmp[:, 1-plan.padup:0], dims = 2)
    # tmp[:, plan.nz:plan.nz] .+= sum(tmp[:, plan.nz+1:end], dims = 2)
    # return tmp[1:plan.nx, 1:plan.nz]
    imfilter!(plan.workimg, plan.padzero(img), plan.psfs[:, :, i, viewidx], NoPad(), plan.alg)
    plan.workimg[1:1, :] .+= sum(plan.workimg[plan.nx+plan.padright+1:end, :], dims = 1)
    plan.workimg[plan.nx:plan.nx, :] .+= sum(plan.workimg[plan.nx+1:plan.nx+plan.padright, :], dims = 1)
    plan.workimg[:, 1:1] .+= sum(plan.workimg[:, plan.nz+plan.paddown+1:end], dims = 2)
    plan.workimg[:, plan.nz:plan.nz] .+= sum(plan.workimg[:, plan.nz + 1:plan.nz + plan.paddown], dims = 2)
    return plan.workimg[1:plan.nx, 1:plan.nz]
end
"""
    my_conv!(output, img, ker, plan)
    Convolve an image with a kernel using in-place operation
    This part is still under construction
    centered(img) should be preallocated, get rid of if
"""
function my_conv!(output, img, ker, plan)
    imfilter!(plan.workimg, plan.mypad!(img), centered(ker), plan.alg)
    return map!(output, x -> max.(x, 0), @view plan.workimg[1:plan.nx, 1:plan.nz])
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
    # todo : read multiple dispatch
    # rotate = x -> my_rotate(x, plan.viewangle[viewidx], plan.interphow)
    # myrotate = x -> imrotate3emmt(x, plan.viewangle[viewidx], plan.nx, plan.ny; mode = :forward)
    # rotate image
    for z = 1:plan.nz
        plan.imgr[:, :, z] = plan.rotateforw(plan.pad_imgr, image[:,:,z],
                                    plan.viewangle[viewidx], plan.nx, plan.ny,
                                    plan.pad_rotate_x, plan.pad_rotate_y)
        plan.mumapr[:, :, z] = plan.rotateforw(plan.pad_imgr, plan.mumap[:,:,z],
                                    plan.viewangle[viewidx], plan.nx, plan.ny,
                                    plan.pad_rotate_x, plan.pad_rotate_y)
    end
    # rotate mumap
    # plan.mumapr = mapslices(x -> plan.rotateforw(x, plan.viewangle[viewidx], plan.nx, plan.ny),
    #                 plan.mumap, dims = [1, 2])
    # loop over image planes
        # use zero-padded fft (because big) or conv (if small) to convolve with psf
        # sum, account for mumap

    for i = 1:plan.ny
        # 0.5 account for the slice thickness
        exp_mumapr = dropdims(exp.(-plan.dy*(sum(plan.mumapr[:, 1:i, :], dims = 2) .- (plan.mumapr[:,i:i,:]/2))); dims = 2)
        view .+= my_conv(plan.imgr[:, i, :] .* exp_mumapr, plan.psfs[:, :, i, viewidx], plan, i, viewidx)
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

    # todo : read multiple dispatch
    # rotate = x -> my_rotate(x, plan.viewangle[viewidx], plan.interphow)

    # myrotate = x -> imrotate3emmt(x, plan.viewangle[viewidx], plan.nx, plan.ny; mode = :forward)
    # myderotate = x -> imrotate3emmt(x, plan.viewangle[viewidx], plan.nx, plan.ny; mode = :adjoint)

    # rotate mumap
    for z = 1:plan.nz
        plan.mumapr[:,:,z] = plan.rotateforw(plan.pad_imgr, plan.mumap[:,:,z],
                                plan.viewangle[viewidx], plan.nx, plan.ny,
                                plan.pad_rotate_x, plan.pad_rotate_y)
    end
    # plan.mumapr = mapslices(x -> plan.rotateforw(x, plan.viewangle[viewidx], plan.nx, plan.ny),
    #                 plan.mumap, dims = [1, 2])

    # adjoint of sum along y axis
    # plan.imgr = repeat(reshape(view, plan.nx, 1, plan.nz), 1, plan.ny, 1)
    imgr = repeat(reshape(view, plan.nx, 1, plan.nz), 1, plan.ny, 1)

    for i = 1:plan.ny
        exp_mumapr = dropdims(exp.(-plan.dy*(sum(plan.mumapr[:, 1:i, :], dims = 2) .- (plan.mumapr[:,i:i,:]/2))); dims = 2) # nx * nz
        # adjoint of convolution, convolve with reverse of psfs
        # adjoint of multiplying with mumap
        imgr[:, i, :] .= my_conv_adj(imgr[:, i, :], plan.psfs[:, :, i, viewidx], plan, i, viewidx) .* exp_mumapr
    end

    # adjoint of imrotate
    for z = 1:plan.nz
        imgr[:,:,z] = plan.rotateadjt(plan.pad_imgr, imgr[:,:,z],
                                plan.viewangle[viewidx], plan.nx, plan.ny,
                                plan.pad_rotate_x, plan.pad_rotate_y)
    end
    return imgr
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
