
const RealU = Number # Union{Real,Unitful.Length}

export rotate_x, rotate_x!


"""
    rotate_x!(out, img, θ, M, N)
Mutating version of `rotate_x`.
"""
function rotate_x!(out, img, θ::RealU, M::Int, N::Int)
    xi = -(M-1)/2 : (M-1)/2
    yi = -(N-1)/2 : (N-1)/2
    rotate_x(xin, yin) = xin + yin * tan(θ/2)
    for (i, yin) in enumerate(yi)
        ic = LinearInterpolation(xi, img[:, i], extrapolation_bc = 0)
        out[:, i] .= ic.(rotate_x.(xi, yin))
    end
    return out
end


"""
    out = rotate_x(img, θ, M, N)
Rotate an image along x axis in clockwise direction using linear interpolation
"""
function rotate_x(out, img::AbstractMatrix{T}, θ, M::Int, N::Int) where T
    out = zeros(T, M, N)
    return rotate_x!(out, img, θ, M, N)
end


"""
    rotate_y(img, θ, M, N)
    rotate an image along y axis in clockwise direction using linear interpolation
"""
function rotate_y(img, θ, M, N)
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
    imrotate3(img, θ, M, N)
    rotate an image by angle θ (must be ranging from 0 to 2π) in clockwise direction using linear interpolation
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
    imrotate3jl(image, θ, plan)
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
