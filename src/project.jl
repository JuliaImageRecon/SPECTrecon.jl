# project.jl
using Interpolations
using ImageTransformations
using ImageFiltering
using OffsetArrays
using FFTW
# myzeropad1 = (x, p) -> padarray(x, Fill(0, (0, 0), (p - size(x, 1), 0))) # zero pad for x
# myreplicatepad2 = (x, p) -> padarray(x, Pad(:replicate, (0, 0), (0, p - size(x, 2)))) # replicate pad for z
Power2 = x -> 2^ceil(Int, log2(x))
"""
    SPECTplan

Struct for storing key factors for a SPECT system model
- `mumap [nx,ny,nz]` attenuation map, must be 3D, possibly zeros()
- `psfs [nx_psf,nz_psf,ny,nview]` must be 4D, with `nx_psf` and `nz_psf` odd, and symmetric for each slice
- `nview` number of views, must be integer
- `interphow` Interpolation method, default is bilinear interpolation
- `viewangle` a vector of angles ranging from 0 to 2π
- `dy` voxel size in y direction (dx is the same value)
- `nx` number of voxels in x direction of the image, must be integer
- `ny` number of voxels in y direction of the image, must be integer
- `nz` number of voxels in z direction of the image, must be integer
- `nx_psf` number of voxels in x direction of the psf, must be integer
- `nz_psf` number of voxels in z direction of the psf, must be integer
- `pad{up,down,left,right}` number of padding voxels in each direction, must be integer
- `mypad` padding function using replicate border condition
Currently code assumes each of the `nview` projection views is `[nx,nz]`
Currently code assumes `nx = ny`
Currently code assumes uniform angular sampling
"""
struct SPECTplan
    mumap::AbstractArray{<:Real} # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:Real} # PSFs could be 3D or 4D
    nview::Int
    interphow::BSpline{<:Any}
    viewangle::AbstractVector
    dy::Float32
    nx::Int
    ny::Int
    nz::Int
    nx_psf::Int
    nz_psf::Int
    padup::Int
    paddown::Int
    padleft::Int
    padright::Int
    mypad::Function
    # other options for how to do the projection?
    function SPECTplan(mumap, psfs, nview, dy; interpidx::Int = 1)
        # check nx = ny ? typically 128 x 128 x 81
        nx, ny, nz = size(mumap)
        nx_psf = size(psf, 1)
        nz_psf = size(psf, 2)
        @assert isequal(nx, ny)
        if interpidx === 0
            interphow = BSpline(Constant()) # nearest neighbor interpolation
        elseif interpidx === 1
            interphow = BSpline(Linear()) # (multi)linear interpolation
        elseif interpidx === 3
            interphow = BSpline(Cubic(Line(OnGrid()))) # cubic b-spline interpolation
        else
            throw("unknown interpidx!")
        end
        # todo: check that nx_psf and nz_psf are odd and very each psf is symmetric
        viewangle = (0:nview-1) / nview * (2π)
        padleft = Int(ceil((Power2(nx+nx_psf-1) - nx) / 2))
        padright = Int(floor((Power2(nx+nx_psf-1) - nx) / 2))
        padup = Int(ceil((Power2(nz+nz_psf-1) - nz) / 2))
        paddown = Int(floor((Power2(nz+nz_psf-1) - nz) / 2))
        mypad = x -> padarray(x, Pad(:replicate, (padleft, padup), (padright, paddown)))
        new(mumap, psfs, nview, interphow, viewangle, dy, nx, ny, nz, nx_psf, nz_psf,
            padup, paddown, padleft, padright, mypad)
        #  creates objects of the block's type (inner constructor methods).
    end
end


"""
    my_conv(img, ker, plan)
    Convolve an image with a kernel using FFT with zero padding
"""
function my_conv(img, ker, plan)
    return max.(0, imfilter(plan.mypad(img), centered(reverse(ker)), Algorithm.FFT()))[1:plan.nx, 1:plan.nz]
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
    rotate = x -> OffsetArrays.no_offset_view(
                            imrotate(x, - plan.viewangle[viewidx], axes(x), 0, # details check the rotation center
                            method = plan.interphow))
    # rotate image
    imgr = mapslices(rotate, image, dims = [1, 2])
    # rotate mumap
    mumapr = mapslices(rotate, plan.mumap, dims = [1, 2])
    # loop over image planes
        # use zero-padded fft (because big) or conv (if small) to convolve with psf
        # sum, account for mumap
    for i = 1:ny
        exp_mumapr = dropdims(exp.(-sum(plan.dy * mumapr[:, 1:i, :], dims = 2)); dims = 2) # nx * nz
        view += my_conv(imgr[:, i, :] .* exp_mumapr, plan.psfs[:, :, i, viewidx], plan)
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
        # project!((@view views[:,:,i]), plan, image, i) # this line doesn't work, the output is all zero
        views[:,:,i] = project!(views[:,:,i], plan, image, i)
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
    nx = size(image, 1)
    nz = size(image, 3)
    views = zeros(promote_type(eltype(image), Float32), nx, nz, plan.nview)
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
    plan = SPECTplan(mumap, psfs, nview, dy; interpidx = interpidx)
    project(plan, image; kwargs...)
end
