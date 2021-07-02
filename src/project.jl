# project.jl
using Interpolations, ImageTransformations, ImageFiltering, OffsetArrays
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
"""
struct SPECTplan
    mumap::AbstractArray{<:Real, 3} # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:Real, 4} # PSFs must be 4D
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
                        conv_alg::Symbol = :fft)
        # check nx = ny ? typically 128 x 128 x 81
        nx, ny, nz = size(mumap)
        nx_psf = size(psfs, 1)
        nz_psf = size(psfs, 2)
        @assert isequal(nx, ny)
        @assert iseven(nx) && iseven(ny)
        @assert isodd(nx_psf) && isodd(nz_psf)
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
        padleft = ceil(Int, (Power2(nx+nx_psf-1) - nx) / 2)
        padright = floor(Int, (Power2(nx+nx_psf-1) - nx) / 2)
        padup = ceil(Int, (Power2(nz+nz_psf-1) - nz) / 2)
        paddown = floor(Int, (Power2(nz+nz_psf-1) - nz) / 2)
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
    if issymmetric(ker)
        return max.(0, imfilter(plan.mypad(img), centered(ker), plan.alg))[1:plan.nx, 1:plan.nz]
    else
        throw("psf is not symmetric!")
    end
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
    rotate_x(img, θ)
    rotate an image along x axis in clockwise direction using linear interpolation
"""
function rotate_x(img, θ)
    M, N = size(img)
    xi = -(M-1)/2 : (M-1)/2
    yi = -(N-1)/2 : (N-1)/2
    rotate_x(xin, yin, θ) = xin + yin * tan(θ/2)
    tmp = zeros(eltype(img), M, N)
    for (i, yin) in enumerate(yi)
        ic = LinearInterpolation(xi, img[:, i], extrapolation_bc = 0)
        tmp[:, i] .= ic.(rotate_x.(xi, yin, θ))
    end
    return tmp
end
"""
    rotate_y(img, θ)
    rotate an image along y axis in clockwise direction using linear interpolation
"""
function rotate_y(img, θ)
    M, N = size(img)
    xi = -(M-1)/2 : (M-1)/2
    yi = -(N-1)/2 : (N-1)/2
    rotate_y(xin, yin, θ) = xin * (-sin(θ)) + yin
    tmp = zeros(eltype(img), M, N)
    for (i, xin) in enumerate(xi)
        ic = LinearInterpolation(yi, img[i, :], extrapolation_bc = 0)
        tmp[i, :] .= ic.(rotate_y.(xin, yi, θ))
    end
    return tmp
end
"""
    rot_back(img, m)
"""
function rot_f90(img, m)
    if m == 0
        return img
    elseif m == 1
        return rotl90(img)
    elseif m == 2
        return rot180(img)
    elseif m == 3
        return rotr90(img)
    else
        throw("invalid m!")
    end
end
"""
    my_rotate_v2(img, θ)
    rotate an image by angle θ (must be ranging from 0 to 2π) in clockwise direction using linear interpolation
"""
function my_rotate_v2(img, θ)
    M, N = size(img)
    m = mod(floor(Int, (θ + π/4) / (π/2)), 4)
    mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    return rotate_x(rotate_y(rotate_x(rot_f90(OffsetArrays.no_offset_view(padarray(img, Pad(:reflect, pad_x, pad_y))), m),
                mod_theta), mod_theta), mod_theta)[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
end
"""
    my_rotate(image, θ, plan)
    Rotate an image by angle θ in counter clockwise direction using built-in imrotate function
"""
function my_rotate(img, θ, interphow)
    if mod(θ, 2π) ≈ 0
        return img
    elseif mod(θ, 2π) ≈ π
        return reverse(img)
    else
        return OffsetArrays.no_offset_view(imrotate(img,
                                            -θ, # rotate angle
                                            axes(img), # crop the image
                                            0, # extrapolation_bc = 0
                                            method = interphow))
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
    rotate = x -> my_rotate_v2(x, plan.viewangle[viewidx])
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
