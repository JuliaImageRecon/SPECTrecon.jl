# rotate3.jl
# todo: RotatePlan
# set of thetas, workspace, choice of rotation method (3 pass, imrotate, emmt?)
# I won't use imrotate3jl because it cannot pass the adjoint test
"""
    rotate_x!(output, img, θ, xi, yi)
    rotate an image along x axis in clockwise direction using 1d linear interpolation,
    storing results in `output`
"""
function rotate_x!(output, img, θ, xi, yi)
    rotate_x(xin, yin, θ) = xin + (yin - (length(yi)+1)/2) * tan(θ/2)
    for (i, yin) in enumerate(yi)
        # note for future refinement: the rotate_x. step is allocating
        A = SparseInterpolator(LinearSpline(Float32), rotate_x.(xi, yin, θ), length(xi))
        # ic = LinearInterpolation(xi, img[:, i], extrapolation_bc = 0)
        # tmp[:, i] .= ic.(rotate_x.(xi, yin, θ))
        mul!((@view output[:, i]), A, img[:, i]) # need mul! to avoid allocating
    end
    return output
end
"""
    rotate_x_adj!(output, img, θ, xi, yi)
    The adjoint of rotating an image along x axis in clockwise direction using 1d linear interpolation,
    storing results in `output`
"""
function rotate_x_adj!(output, img, θ, xi, yi)
    rotate_x(xin, yin, θ) = xin + (yin - (length(yi)+1)/2) * tan(θ/2)
    for (i, yin) in enumerate(yi)
        A = SparseInterpolator(LinearSpline(Float32), rotate_x.(xi, yin, θ), length(xi))
        # ic = LinearInterpolation(xi, img[:, i], extrapolation_bc = 0)
        # tmp[:, i] .= ic.(rotate_x.(xi, yin, θ))
        mul!((@view output[:, i]), A', img[:, i])
        # @! output[:, i] .= I1' * img[:, i]
    end
    return output
end

"""
    rotate_y!(output, img, θ, xi, yi)
    rotate an image along y axis in clockwise direction using 1d linear interpolation,
    storing results in `output`
"""
function rotate_y!(output, img, θ, xi, yi)
    rotate_y(xin, yin, θ) = (xin - (length(xi)+1)/2) * (-sin(θ)) + yin
    for (i, xin) in enumerate(xi)
        A = SparseInterpolator(LinearSpline(Float32), rotate_y.(xin, yi, θ), length(yi))
        mul!((@view output[i, :]), A, img[i, :])
        # ic = LinearInterpolation(yi, img[i, :], extrapolation_bc = 0)
        # tmp[i, :] .= ic.(rotate_y.(xin, yi, θ))
    end
    return output
end

"""
    rotate_y_adj!(output, img, θ, xi, yi)
    The adjoint of rotating an image along y axis in clockwise direction using 1d linear interpolation,
    storing results in `output`
"""
function rotate_y_adj!(output, img, θ, xi, yi)
    rotate_y(xin, yin, θ) = (xin - (length(xi)+1)/2) * (-sin(θ)) + yin
    for (i, xin) in enumerate(xi)
        A = SparseInterpolator(LinearSpline(Float32), rotate_y.(xin, yi, θ), length(yi))
        mul!((@view output[i, :]), A', img[i, :])
        # ic = LinearInterpolation(yi, img[i, :], extrapolation_bc = 0)
        # tmp[i, :] .= ic.(rotate_y.(xin, yi, θ))
    end
    return output
end

"""
    rotl90!(B::AbstractMatrix, A::AbstractMatrix)
    In place version of `rotl90`, returning rotation of `A` in `B`.
"""
function rotl90!(B::AbstractMatrix, A::AbstractMatrix)
    ind1, ind2 = axes(A)
    n = first(ind2) + last(ind2)
    for i = axes(A, 1), j = ind2
        B[n - j, i] = A[i, j]
    end
    return B
end

"""
    rotr90!(B::AbstractMatrix, A::AbstractMatrix)
    In place version of `rotr90`, returning rotation of `A` in `B`.
"""
function rotr90!(B::AbstractMatrix, A::AbstractMatrix)
    ind1, ind2 = axes(A)
    m = first(ind1) + last(ind1)
    for i = ind1, j = axes(A, 2)
        B[j, m - i] = A[i, j]
    end
    return B
end

"""
    rot180!(B::AbstractMatrix, A::AbstractMatrix)
    In place version of `rot180`, returning rotation of `A` in `B`.
"""
function rot180!(B::AbstractMatrix, A::AbstractMatrix)
    ind1, ind2 = axes(A,1), axes(A,2)
    m, n = first(ind1)+last(ind1), first(ind2)+last(ind2)
    for j=ind2, i=ind1
        B[m-i,n-j] = A[i,j]
    end
    return B
end

"""
    rot_f90!(output, img, m)
    rotate an image by 90/180/270 degrees
    Note: these are all allocating.  Future work is to add rot180! etc. to Julia
"""
function rot_f90!(output, img, m)
    if m == 0
        return copyto!(output, img)
    elseif m == 1
        return rotl90!(output, img)
    elseif m == 2
        return rot180!(output, img)
    elseif m == 3
        return rotr90!(output, img)
    else
        throw("invalid m!")
    end
end

"""
    rot_f90_adj!(output, img, m)
    The adjoint of rotating an image by 90/180/270 degrees
"""
function rot_f90_adj!(output, img, m)
    if m == 0
        return copyto!(output, img)
    elseif m == 1
        return rotr90!(output, img)
    elseif m == 2
        return rot180!(output, img)
    elseif m == 3
        return rotl90!(output, img)
    else
        throw("invalid m!")
    end
end
"""
    imrotate3!(output, tmp, img, θ, M, N, pad_x, pad_y)
    rotate an image by angle θ (must be ranging from 0 to 2π) in clockwise direction using a series of 1d linear interpolation
"""
function imrotate3!(output, tmp, img, θ, M, N, pad_x, pad_y)
    m = mod(floor(Int, 0.5 + θ/(π/2)), 4)
    mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
    xi = 1 : M + 2 * pad_x
    yi = 1 : N + 2 * pad_y
    tmp .= OffsetArrays.no_offset_view(BorderArray(img, Fill(0, (pad_x, pad_y))))
    rot_f90!(output, tmp, m)
    rotate_x!(output, output, mod_theta, xi, yi)
    rotate_y!(output, output, mod_theta, xi, yi)
    rotate_x!(output, output, mod_theta, xi, yi)
    return @view output[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
end

"""
    imrotate3_adj!(output, tmp, img, θ, M, N, pad_x, pad_y)
    The adjoint of rotating an image by angle θ (must be ranging from 0 to 2π) in clockwise direction using a series of 1d linear interpolation
"""
function imrotate3_adj!(output, tmp, img, θ, M, N, pad_x, pad_y)
    m = mod(floor(Int, 0.5 + θ/(π/2)), 4)
    mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
    xi = 1 : M + 2 * pad_x
    yi = 1 : N + 2 * pad_y
    tmp .= OffsetArrays.no_offset_view(BorderArray(img, Fill(0, (pad_x, pad_y))))
    rotate_x_adj!(tmp, tmp, mod_theta, xi, yi)
    rotate_y_adj!(tmp, tmp, mod_theta, xi, yi)
    rotate_x_adj!(tmp, tmp, mod_theta, xi, yi)
    rot_f90_adj!(output, tmp, m) # must be two different arguments
    return @view output[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
end


"""
    imrotate3emmt!(output, tmp, img, θ, M, N, pad_x, pad_y)
    Rotate an image by angle θ in counter clockwise direction using 2d linear interpolation
    Source code is here: https://github.com/emmt/LinearInterpolators.jl
"""
function imrotate3emmt!(output, tmp, img, θ, M, N, pad_x, pad_y)
    if mod(θ, 2π) ≈ 0
        return img
    else
        ker = LinearSpline(Float32)
        M_pad = M + 2 * pad_x
        N_pad = N + 2 * pad_y
        rows = (M_pad, N_pad)
        cols = (M_pad, N_pad)
        c = ((1 + M_pad) / 2, (1 + N_pad) / 2)
        R = c + rotate(2π - θ, AffineTransform2D{Float32}() - c)
        A = TwoDimensionalTransformInterpolator(rows, cols, ker, R)
        tmp .= OffsetArrays.no_offset_view(BorderArray(img, Fill(0, (pad_x, pad_y))))
        mul!(output, A, tmp)
        return @view output[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
    end
end

"""
    imrotate3emmt_adj!(output, tmp, img, θ, M, N, pad_x, pad_y)
    The adjoint of rotating an image by angle θ in counter clockwise direction using 2d linear interpolation
    Source code is here: https://github.com/emmt/LinearInterpolators.jl
"""
function imrotate3emmt_adj!(output, tmp, img, θ, M, N, pad_x, pad_y)
    if mod(θ, 2π) ≈ 0
        return img
    else
        ker = LinearSpline(Float32)
        M_pad = M + 2 * pad_x
        N_pad = N + 2 * pad_y
        rows = (M_pad, N_pad)
        cols = (M_pad, N_pad)
        c = ((1 + M_pad) / 2, (1 + N_pad) / 2)
        R = c + rotate(2π - θ, AffineTransform2D{Float32}() - c)
        A = TwoDimensionalTransformInterpolator(rows, cols, ker, R)
        tmp .= OffsetArrays.no_offset_view(BorderArray(img, Fill(0, (pad_x, pad_y))))
        mul!(output, A', tmp)
        return @view output[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
    end
end

# """
#     imrotate3jl(img, θ, plan)
#     Rotate an image by angle θ in counter clockwise direction using "Julia" built-in imrotate function
#     Not used because it cannot pass the adjoint test
# """
# function imrotate3jl(img, θ, interphow)
#     if mod(θ, 2π) ≈ 0
#         return img
#     elseif mod(θ, 2π) ≈ π
#         return rot180(img)
#     else
#         return OffsetArrays.no_offset_view(imrotate(img,
#                                             -θ, # rotate angle
#                                             axes(img), # crop the image
#                                             0, # extrapolation_bc = 0
#                                             method = interphow))
#     end
# end

# Some test code
# M = 16
# N = 16
# pad_x = 8
# pad_y = 8
# xi = 1 : M + 2 * pad_x
# yi = 1 : N + 2 * pad_y
# θ = π/7
# x = randn(Float32, M, N)
# y = randn(Float32, M, N)
# output_x = OffsetArrays.no_offset_view(padarray(x, Fill(0, (pad_x, pad_y))))
# output_y = OffsetArrays.no_offset_view(padarray(y, Fill(0, (pad_x, pad_y))))
# isapprox(vdot(y, imrotate3emmt!(output_x, x, θ, M, N, pad_x, pad_y)),
#             vdot(x, imrotate3emmt_adj!(output_y, y, θ, M, N, pad_x, pad_y)))

# img = zeros(Float32, M, N)
# img[4:13, 4:13] .= 1
# output = OffsetArrays.no_offset_view(padarray(img, Fill(0, (pad_x, pad_y))))
# tmp = similar(output)
# jim(imrotate3_adj!(output, tmp, img, θ, M, N, pad_x, pad_y))
# jim(imrotate3!(output, tmp, img, θ, M, N, pad_x, pad_y))
# jim(imrotate3!(output, img, θ, M, N, pad_x, pad_y) - imrotate3emmt!(output, img, θ, M, N, pad_x, pad_y))
