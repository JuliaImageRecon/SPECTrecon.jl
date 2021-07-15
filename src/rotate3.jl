using LinearAlgebra
using LazyAlgebra, TwoDimensional
using LinearInterpolators
using InterpolationKernels
using OffsetArrays
using ImageFiltering
# todo: RotatePlan
# set of thetas, workspace, choice of rotation method (3 pass, imrotate, emmt?)


"""
    rotate_x(output, img, θ, xi, yi)
    rotate an image along x axis in clockwise direction using 1d linear interpolation
"""
function rotate_x!(output, img, θ, xi, yi)
    rotate_x(xin, yin, θ) = xin + (yin - (length(yi)+1)/2) * tan(θ/2)
    for (i, yin) in enumerate(yi)
        I1 = SparseInterpolator(LinearSpline(Float32), rotate_x.(xi, yin, θ), length(xi))
        # ic = LinearInterpolation(xi, img[:, i], extrapolation_bc = 0)
        # tmp[:, i] .= ic.(rotate_x.(xi, yin, θ))
        @! output[:, i] .= I1 * img[:, i] # need mul! to avoid allocating
    end
    return output
end
"""
    rotate_x_adj(output, img, θ, xi, yi)
    The adjoint of rotating an image along x axis in clockwise direction using 1d linear interpolation
"""
function rotate_x_adj!(output, img, θ, xi, yi)
    rotate_x(xin, yin, θ) = xin + (yin - (length(yi)+1)/2) * tan(θ/2)
    for (i, yin) in enumerate(yi)
        I1 = SparseInterpolator(LinearSpline(Float32), rotate_x.(xi, yin, θ), length(xi))
        # ic = LinearInterpolation(xi, img[:, i], extrapolation_bc = 0)
        # tmp[:, i] .= ic.(rotate_x.(xi, yin, θ))
        output[:, i] .= I1' * img[:, i]
    end
    return output
end

"""
    rotate_y(output, img, θ, xi, yi)
    rotate an image along y axis in clockwise direction using 1d linear interpolation
"""
function rotate_y!(output, img, θ, xi, yi)
    rotate_y(xin, yin, θ) = (xin - (length(xi)+1)/2) * (-sin(θ)) + yin
    for (i, xin) in enumerate(xi)
        I1 = SparseInterpolator(LinearSpline(Float32), rotate_y.(xin, yi, θ), length(yi))
        output[i, :] .= I1 * img[i, :]
        # ic = LinearInterpolation(yi, img[i, :], extrapolation_bc = 0)
        # tmp[i, :] .= ic.(rotate_y.(xin, yi, θ))
    end
    return output
end

"""
    rotate_y_adj!(output, img, θ, xi, yi)
    The adjoint of rotating an image along y axis in clockwise direction using 1d linear interpolation
"""
function rotate_y_adj!(output, img, θ, xi, yi)
    rotate_y(xin, yin, θ) = (xin - (length(xi)+1)/2) * (-sin(θ)) + yin
    for (i, xin) in enumerate(xi)
        I1 = SparseInterpolator(LinearSpline(Float32), rotate_y.(xin, yi, θ), length(yi))
        output[i, :] .= I1' * img[i, :]
        # ic = LinearInterpolation(yi, img[i, :], extrapolation_bc = 0)
        # tmp[i, :] .= ic.(rotate_y.(xin, yi, θ))
    end
    return output
end
"""
    rot_f90(img, m)
    rotate an image by 90/180/270 degrees
    Note: these are all allocating.  Future work is to add rot180! etc. to Julia
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
    rot_f90_adj(img, m)
    The adjoint of rotating an image by 90/180/270 degrees
"""
function rot_f90_adj(img, m)
    if m == 0
        return img
    elseif m == 1
        return rotr90(img)
    elseif m == 2
        return rot180(img)
    elseif m == 3
        return rotl90(img)
    else
        throw("invalid m!")
    end
end
"""
    imrotate3(img, θ, M, N)
    rotate an image by angle θ (must be ranging from 0 to 2π) in clockwise direction using a series of 1d linear interpolation
"""
function imrotate3!(output, img, θ, M, N, pad_x, pad_y)
    m = mod(floor(Int, 0.5 + θ/(π/2)), 4)
    mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
    xi = 1 : M + 2 * pad_x
    yi = 1 : N + 2 * pad_y
    img = rot_f90(OffsetArrays.no_offset_view(BorderArray(img, Fill(0, (pad_x, pad_y)))), m)
    rotate_x!(output, img, mod_theta, xi, yi)
    rotate_y!(output, output, mod_theta, xi, yi)
    rotate_x!(output, output, mod_theta, xi, yi)
    return output[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
end

"""
    imrotate3_adj(img, θ, M, N)
    The adjoint of rotating an image by angle θ (must be ranging from 0 to 2π) in clockwise direction using a series of 1d linear interpolation
"""
function imrotate3_adj!(output, img, θ, M, N, pad_x, pad_y)
    m = mod(floor(Int, 0.5 + θ/(π/2)), 4)
    mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
    xi = 1 : M + 2 * pad_x
    yi = 1 : N + 2 * pad_y
    img = OffsetArrays.no_offset_view(BorderArray(img, Fill(0, (pad_x, pad_y))))
    rotate_x_adj!(output, img, mod_theta, xi, yi)
    rotate_y_adj!(output, output, mod_theta, xi, yi)
    rotate_x_adj!(output, output, mod_theta, xi, yi)
    return rot_f90_adj(output, m)[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
end

"""
    imrotate3jl(img, θ, plan)
    Rotate an image by angle θ in counter clockwise direction using "Julia" built-in imrotate function
    Not used in this version
"""
function imrotate3jl(img, θ, interphow)
    if mod(θ, 2π) ≈ 0
        return img
    elseif mod(θ, 2π) ≈ π
        return rot180(img)
    else
        return OffsetArrays.no_offset_view(imrotate(img,
                                            -θ, # rotate angle
                                            axes(img), # crop the image
                                            0, # extrapolation_bc = 0
                                            method = interphow))
    end
end
"""
    imrotate3emmt!(output, img, θ, M, N, pad_x, pad_y)
    Rotate an image by angle θ in counter clockwise direction using 2d linear interpolation
    Source code is here: https://github.com/emmt/LinearInterpolators.jl
"""
function imrotate3emmt!(output, img, θ, M, N, pad_x, pad_y)
    if mod(θ, 2π) ≈ 0
        return img
    elseif mod(θ, 2π) ≈ π
        return rot180(img)
    else
        output = OffsetArrays.no_offset_view(BorderArray(img, Fill(0, (pad_x, pad_y))))
        ker = LinearSpline(Float32)
        M_pad = M + 2 * pad_x
        N_pad = N + 2 * pad_y
        rows = (M_pad, N_pad)
        cols = (M_pad, N_pad)
        c = ((1 + M_pad) / 2, (1 + N_pad) / 2)
        R = c + rotate(2π - θ, AffineTransform2D{Float32}() - c)
        A = TwoDimensionalTransformInterpolator(rows, cols, ker, R)
        return (A * output)[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
    end
end

"""
    imrotate3emmt_adj!(output, img, θ, M, N, pad_x, pad_y)
    The adjoint of rotating an image by angle θ in counter clockwise direction using 2d linear interpolation
    Source code is here: https://github.com/emmt/LinearInterpolators.jl
"""
function imrotate3emmt_adj!(output, img, θ, M, N, pad_x, pad_y)
    if mod(θ, 2π) ≈ 0
        return copyto!(output, img)
    elseif mod(θ, 2π) ≈ π
        return rot180!(output, img)
    else
        output = OffsetArrays.no_offset_view(BorderArray(img, Fill(0, (pad_x, pad_y))))
        ker = LinearSpline(Float32)
        M_pad = M + 2 * pad_x
        N_pad = N + 2 * pad_y
        rows = (M_pad, N_pad)
        cols = (M_pad, N_pad)
        c = ((1 + M_pad) / 2, (1 + N_pad) / 2)
        R = c + rotate(2π - θ, AffineTransform2D{Float32}() - c)
        A = TwoDimensionalTransformInterpolator(rows, cols, ker, R)
        return (A' * output)[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
    end
end

"""
    rot180!(B::AbstractMatrix, A::AbstractMatrix)
In place version of `rot180`, returning rotation of `A` in `B`.
"""
function rot180!(B::AbstractMatrix, A::AbstractMatrix)
    ind1, ind2 = axes(A,1), axes(A,2)
    m, n = first(ind1)+last(ind1), first(ind2)+last(ind2)
    for j=ind2, i=ind1
        B[m-i,n-j] .= A[i,j]
    end
    return B
end
