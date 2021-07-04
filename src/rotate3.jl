"""
    rotate_x(img, θ, M, N)
    rotate an image along x axis in clockwise direction using linear interpolation
"""
function rotate_x(img, θ, M, N)
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
    return rotate_x(rotate_y(rotate_x(rot_f90(OffsetArrays.no_offset_view(padarray(img, Fill(0, (pad_x, pad_y)))), m),
                mod_theta, M, N), mod_theta, M, N), mod_theta, M, N)[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
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
