# project.jl
using Interpolations, ImageTransformations, ImageFiltering, OffsetArrays, LinearAlgebra
include("rotate3.jl")
"""
    SPECTplan

Struct for storing key factors for a SPECT system model
- `mumap [nx,ny,nz]` attenuation map, must be 3D, possibly zeros()
- `psfs [nx_psf,nz_psf,ny,nview]` point spread function, must be 4D, with `nx_psf` and `nz_psf` odd, and symmetric for each slice
- `nview` number of views, must be integer
- `interphow` Interpolation method, default is bilinear interpolation
- `viewangle` a vector of angles ranging from 0 to 2π
- `dy` voxel size in y direction (dx is the same value)
- `nx` number of voxels in x direction of the image, must be integer
- `ny` number of voxels in y direction of the image, must be integer
- `nz` number of voxels in z direction of the image, must be integer
- `nx_psf` number of voxels in x direction of the psf, must be integer
- `nz_psf` number of voxels in z direction of the psf, must be integer
- `mypad` padding function using replicate border condition
- `alg` algorithms used for convolution, default is FFT
Currently code assumes each of the `nview` projection views is `[nx,nz]`
Currently code assumes `nx = ny`
Currently code assumes uniform angular sampling
Currently imrotate3 code only supports linear interpolation
"""
struct SPECTplan
    mumap::AbstractArray{<:Real, 3} # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:Real, 4} # PSFs must be 4D, [nx_psf, nz_psf, ny, nview]
    nview::Int
    interphow::BSpline{<:Any}
    viewangle::AbstractVector
    dy::Float32
    nx::Int
    ny::Int
    nz::Int
    nx_psf::Int
    nz_psf::Int
    mypad::Function
    alg::Any
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
        @assert all(mapslices(issymmetric, psfs, dims = [1, 2]))
        if interpidx == 0
            interphow = BSpline(Constant()) # nearest neighbor interpolation
        elseif interpidx == 1
            interphow = BSpline(Linear()) # (multi)linear interpolation
        elseif interpidx == 3
            interphow = BSpline(Cubic(Line(OnGrid()))) # cubic b-spline interpolation
        else
            throw("unknown interpidx!")
        end
        viewangle = (0:nview-1) / nview * (2π)

        Power2 = x -> 2^(ceil(Int, log2(x)))

        padleft = iszero(padleft) ? ceil(Int, (Power2(nx+nx_psf-1) - nx) / 2) : padleft
        padright = iszero(padright) ? floor(Int, (Power2(nx+nx_psf-1) - nx) / 2) : padright
        padup = iszero(padup) ? ceil(Int, (Power2(nz+nz_psf-1) - nz) / 2) : padup
        paddown = iszero(paddown) ? floor(Int, (Power2(nz+nz_psf-1) - nz) / 2) : paddown

        mypad = x -> padarray(x, Pad(:replicate, (padleft, padup), (padright, paddown)))

        if conv_alg === :fft
            alg = Algorithm.FFT()
        elseif conv_alg === :fir
            alg = Algorithm.FIR()
        else
            throw("unknown convolution algorithm choice!")
        end
        new(mumap, psfs, nview, interphow, viewangle, dy, nx, ny, nz,
                nx_psf, nz_psf, mypad, alg)
        #  creates objects of the block's type (inner constructor methods).
    end
end


"""
    my_conv(img, ker, plan)
    Convolve an image with a kernel
"""
function my_conv(img, ker, plan)
    return max.(0, imfilter(plan.mypad(img), centered(ker), plan.alg))[1:plan.nx, 1:plan.nz]
end

"""
    my_conv!(output, img, ker, plan)
    Convolve an image with a kernel using in-place operation
    This part is still under construction
"""
function my_conv!(output, img, ker, plan)
    if issymmetric(ker)
        imfilter!(plan.workimg, plan.mypad(img), centered(ker), plan.alg)
        return map!(output, x -> max.(x, 0), @view plan.workimg[1:plan.nx, 1:plan.nz])
    else
        throw("psf is not symmetric!")
    end
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
    rotate = x -> imrotate3(x, plan.viewangle[viewidx], plan.nx, plan.ny)
    # rotate image
    imgr = mapslices(rotate, image, dims = [1, 2])
    # rotate mumap
    mumapr = mapslices(rotate, plan.mumap, dims = [1, 2])
    # loop over image planes
        # use zero-padded fft (because big) or conv (if small) to convolve with psf
        # sum, account for mumap
    for i = 1:plan.ny
        # 0.5 account for the slice thickness
        exp_mumapr = dropdims(exp.(-plan.dy*(sum(mumapr[:, 1:i, :], dims = 2) .- (mumapr[:,i:i,:]/2))); dims = 2) # nx * nz
        view .+= my_conv(imgr[:, i, :] .* exp_mumapr, plan.psfs[:, :, i, viewidx], plan)
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
