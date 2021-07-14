using LazyAlgebra, TwoDimensional
using LinearInterpolators
using InterpolationKernels
"""
    rotate_x(img, θ, M, N)
    rotate an image along x axis in clockwise direction using 1d linear interpolation
"""
function rotate_x(img, θ, M, N)
    # xi = -(M-1)/2 : (M-1)/2
    # yi = -(N-1)/2 : (N-1)/2
    xi = 1:M
    yi = 1:N
    rotate_x(xin, yin, θ) = xin + (yin - (N+1)/2) * tan(θ/2)
    tmp = zeros(eltype(img), M, N)
    for (i, yin) in enumerate(yi)
        I1 = SparseInterpolator(LinearSpline(Float32), rotate_x.(xi, yin, θ), M)
        # ic = LinearInterpolation(xi, img[:, i], extrapolation_bc = 0)
        # tmp[:, i] .= ic.(rotate_x.(xi, yin, θ))
        tmp[:, i] .= I1 * img[:, i]
    end
    return tmp
end
"""
    rotate_x_adj(img, θ, M, N)
    The adjoint of rotating an image along x axis in clockwise direction using 1d linear interpolation
"""
function rotate_x_adj(img, θ, M, N)
    # xi = -(M-1)/2 : (M-1)/2
    # yi = -(N-1)/2 : (N-1)/2
    xi = 1:M
    yi = 1:N
    rotate_x(xin, yin, θ) = xin + (yin - (N+1)/2) * tan(θ/2)
    tmp = zeros(eltype(img), M, N)
    for (i, yin) in enumerate(yi)
        I1 = SparseInterpolator(LinearSpline(Float32), rotate_x.(xi, yin, θ), M)
        # ic = LinearInterpolation(xi, img[:, i], extrapolation_bc = 0)
        # tmp[:, i] .= ic.(rotate_x.(xi, yin, θ))
        tmp[:, i] .= I1' * img[:, i]
    end
    return tmp
end

"""
    rotate_y(img, θ, M, N)
    rotate an image along y axis in clockwise direction using 1d linear interpolation
"""
function rotate_y(img, θ, M, N)
    # xi = -(M-1)/2 : (M-1)/2
    # yi = -(N-1)/2 : (N-1)/2
    xi = 1:M
    yi = 1:N
    rotate_y(xin, yin, θ) = (xin - (M+1)/2) * (-sin(θ)) + yin
    tmp = zeros(eltype(img), M, N)
    for (i, xin) in enumerate(xi)
        I1 = SparseInterpolator(LinearSpline(Float32), rotate_y.(xin, yi, θ), N)
        tmp[i, :] .= I1 * img[i, :]
        # ic = LinearInterpolation(yi, img[i, :], extrapolation_bc = 0)
        # tmp[i, :] .= ic.(rotate_y.(xin, yi, θ))
    end
    return tmp
end

"""
    rotate_y_adj(img, θ, M, N)
    The adjoint of rotating an image along y axis in clockwise direction using 1d linear interpolation
"""
function rotate_y_adj(img, θ, M, N)
    # xi = -(M-1)/2 : (M-1)/2
    # yi = -(N-1)/2 : (N-1)/2
    xi = 1:M
    yi = 1:N
    rotate_y(xin, yin, θ) = (xin - (M+1)/2) * (-sin(θ)) + yin
    tmp = zeros(eltype(img), M, N)
    for (i, xin) in enumerate(xi)
        I1 = SparseInterpolator(LinearSpline(Float32), rotate_y.(xin, yi, θ), N)
        tmp[i, :] .= I1' * img[i, :]
        # ic = LinearInterpolation(yi, img[i, :], extrapolation_bc = 0)
        # tmp[i, :] .= ic.(rotate_y.(xin, yi, θ))
    end
    return tmp
end
"""
    rot_f90(img, m)
    rotate an image by 90/180/270 degrees
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
function imrotate3(img, θ, M, N)
    m = mod(floor(Int, 0.5 + θ/(π/2)), 4)
    mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    M_pad = M + 2 * pad_x
    N_pad = N + 2 * pad_y
    return rotate_x(rotate_y(rotate_x(rot_f90(OffsetArrays.no_offset_view(padarray(img, Fill(0, (pad_x, pad_y)))), m),
                mod_theta, M_pad, N_pad), mod_theta, M_pad, N_pad), mod_theta, M_pad, N_pad)[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
end
"""
    imrotate3_adj(img, θ, M, N)
    The adjoint of rotating an image by angle θ (must be ranging from 0 to 2π) in clockwise direction using a series of 1d linear interpolation
"""
function imrotate3_adj(img, θ, M, N)
    m = mod(floor(Int, 0.5 + θ/(π/2)), 4)
    mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    M_pad = M + 2 * pad_x
    N_pad = N + 2 * pad_y
    return rot_f90_adj(rotate_x_adj(rotate_y_adj(rotate_x_adj(OffsetArrays.no_offset_view(padarray(img, Fill(0, (pad_x, pad_y)))),
                mod_theta, M_pad, N_pad), mod_theta, M_pad, N_pad), mod_theta, M_pad, N_pad), m)[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
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
    imrotate3emmt(img, θ, M, N)
    Rotate an image by angle θ in counter clockwise direction using 2d linear interpolation
    Source code is here: https://github.com/emmt/LinearInterpolators.jl
"""
function imrotate3emmt(img, θ, M, N)
    if mod(θ, 2π) ≈ 0
        return img
    elseif mod(θ, 2π) ≈ π
        return rot180(img)
    else
        pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
        pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
        padded_img = OffsetArrays.no_offset_view(padarray(img, Fill(0, (pad_x, pad_y))))
        ker = LinearSpline(Float32)
        M_pad = M + 2 * pad_x
        N_pad = N + 2 * pad_y
        rows = (M_pad, N_pad)
        cols = (M_pad, N_pad)
        c = ((1 + M_pad) / 2, (1 + N_pad) / 2)
        R = c + rotate(2π - θ, AffineTransform2D{Float32}() - c)
        A = TwoDimensionalTransformInterpolator(rows, cols, ker, R)
        return (A * padded_img)[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
    end
end

"""
    imrotate3emmt_adj(img, θ, M, N)
    The adjoint of rotating an image by angle θ in counter clockwise direction using 2d linear interpolation
    Source code is here: https://github.com/emmt/LinearInterpolators.jl
"""
function imrotate3emmt_adj(img, θ, M, N)
    if mod(θ, 2π) ≈ 0
        return img
    elseif mod(θ, 2π) ≈ π
        return rot180(img)
    else
        pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
        pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
        padded_img = OffsetArrays.no_offset_view(padarray(img, Fill(0, (pad_x, pad_y))))
        ker = LinearSpline(Float32)
        M_pad = M + 2 * pad_x
        N_pad = N + 2 * pad_y
        rows = (M_pad, N_pad)
        cols = (M_pad, N_pad)
        c = ((1 + M_pad) / 2, (1 + N_pad) / 2)
        R = c + rotate(2π - θ, AffineTransform2D{Float32}() - c)
        A = TwoDimensionalTransformInterpolator(rows, cols, ker, R)
        return (A' * padded_img)[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
    end
end
