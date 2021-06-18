# project.jl
using Interpolations
using ImageTransformations
using ImageFiltering
using OffsetArrays
using FFTW
# reale = (x) -> (@assert x ≈ real(x); real(x))
# myfft = x -> fftshift(fft(ifftshift(x)))
# myifft = x -> fftshift(ifft(ifftshift(x)))
mypad = (x, p) -> padarray(x, Fill(0, (0, 0), p .- size(x)))
"""
    SPECTplan

Struct for storing key factors for a SPECT system model
- `mumap [nx,ny,nz]` attenuation map, must be 3D, possibly zeros()
- `psfs [n,n,ny,nview]` usually 4D, but could be 3D for a circular orbit
- `nview` number of views, must be integer
- `interphow` Interpolation methods, default is bilinear interpolation
Currently code assumes each of the `nview` projection views is `[nx,nz]`
Currently code assumes `nx = ny`
"""
struct SPECTplan
    mumap::AbstractArray{<:Real} # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:Real} # PSFs could be 3D or 4D
    nview::Int
    # work_fft::AbstractMatrix
    interphow::BSpline{<:Any}
    # other options for how to do the projection?
    function SPECTplan(mumap, psfs, nview::Int; interpidx::Int = 2)
        # check nx = ny ? typically 128 x 128 x 81
        nx, ny, nz = size(mumap)
        @assert isequal(nx, ny)
        # work_fft = zeros(T, size?)
        if interpidx === 1
            interphow = BSpline(Constant()) # nearest neighbor interpolation
        elseif interpidx === 2
            interphow = BSpline(Linear()) # (multi)linear interpolation
        elseif interpidx === 3
            interphow = BSpline(Cubic(Line(OnGrid()))) # cubic b-spline interpolation
        else
            throw("unknown interpidx!")
        end
        new(mumap, psfs, nview, interphow) #  creates objects of the block's type (inner constructor methods).
    end
end


"""
    my_conv(img, ker)
    Convolve an image with a kernel using FFT with zero padding
"""
function my_conv(img, ker)
    nx, nz = size(img)
    # p, q = size(ker)
    px = 2 * nx
    pz = 2 * nz
    img_padded = mypad(img, (px, pz))
    # kernel is already normalized
    w = centered(reverse(ker))
    # ker_padded = freqkernel(centered(ker), (px, pz))
    # blurred_img_padded = reale(myifft(myfft(img_padded) .* ker_padded))
    # blurred_img = blurred_img_padded[1:nx, 1:nz]
    blurred_img_padded = imfilter(img_padded, w, Fill(0, w), Algorithm.FFT())
    blurred_img = blurred_img_padded[1:nx, 1:nz]
    return blurred_img
end

"""
    1. project!(view, plan, image, viewidx)
    Project a single view
"""
function project!(
    view::AbstractMatrix{<:AbstractFloat},
    plan::SPECTplan,
    image::AbstractArray{<:Real,3},
    viewidx::Int,
)
    # todo : read multiple dispatch

    # rotate image, image is in nx * ny * nz, typically 128 x 128 x 81
    # mumap, psfs, nview, work_fft, interphow = SPECTplan(mumap, psfs, plan.nview, plan.interpidx)
    θ = - 2π / plan.nview * viewidx
    rotate = x -> OffsetArrays.no_offset_view(
                            imrotate(x, θ, axes(x), 0, # details check the rotation center
                            method = plan.interphow))
    imgr = mapslices(rotate, image, dims = [1, 2])
    # rotate mumap
    mumapr = mapslices(rotate, plan.mumap, dims = [1, 2])
    # running sum and exp? of mumap
    # view = zeros(eltype(image), nx, nz) # one projection view
    # No loop is actually needed, maybe using cumsum and mapslices
    # temporary variable, should be part of the plan
    # px = 2 * nx
    # pz = 2 * nz
    # exp_mumapr = exp.(reverse(-cumsum(reverse(mumapr; dims = 2), dims = 2), dims=2)) # nx * ny * nz
    # freqpsf = permutedims(mapslices(x -> freqkernel(centered(x), (px, pz)), plan.psfs[:,:,:,viewidx], dims = [1,2]), [1,3,2]) # px * pz * ny -> px * ny * pz
    # freq_img_padded = mapslices(x -> myfft(mypad(x, (px, pz))), imgr, dims = [1,3]) # px * ny * pz
    # blurred_imgr = mapslices(x -> reale(myifft(x))[1:nx, 1:nz], freq_img_padded .* freqpsf, dims = [1,3]) # nx * ny * nz
    # w = centered(reverse(ker))
    # blurred_imgr = mapslices(x -> imfilter)
    # view = dropdims(sum(blurred_imgr .* exp_mumapr, dims = 2); dims = 2)

    for i = 1:ny
        exp_mumapr = dropdims(exp.(-sum(mumapr[:, 1:i, :], dims = 2)); dims = 2)
        blurred_imgr = my_conv(imgr[:, i, :], plan.psfs[:, :, i, viewidx])
        view += blurred_imgr .* exp_mumapr
    end
    # loop over image planes
        # use zero-padded fft (because big) or conv (if small) to convolve with psf
        # sum, account for mumap
    return view
end


"""
    2. project!(views, plan, image ; index)
    Project multiple views, call 1
"""
function project!(
    views::AbstractArray{<:AbstractFloat,3},
    plan::SPECTplan,
    image::AbstractArray{<:Real,3} ;
    index::AbstractVector{<:Int} = 1:plan.nview, # all views
)

    # loop over each view index
    for i in index
        views[:,:,i] = project!(views[:,:,i], plan, image, i)
    end
    return views
end


"""
    3. views = project(plan, image ; index)
    Initialize views, call 2
"""
function project(
    plan::SPECTplan,
    image::AbstractArray{<:Real,3} ;
    kwargs...,
)
    nx = size(image, 1)
    nz = size(image, 3)
    views = zeros(eltype(image), nx, nz, plan.nview)
    return project!(views, plan, image ; kwargs...)
end


"""
    views = project(image, mumap, psfs, nview, interpidx; kwargs...), test the function, call 3
"""
function project(
    image::AbstractArray{<:Real,3},
    mumap::AbstractArray{<:Real}, # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:Real},
    nview::Int;
    interpidx::Int = 2,
    kwargs...,
)
    plan = SPECTplan(mumap, psfs, nview; interpidx = interpidx)
    project(plan, image ; kwargs...)
end
