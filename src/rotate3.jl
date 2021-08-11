# rotate3.jl
# todo: RotatePlan, use permutedims!, simplify Threads.@threads
# set of thetas, workspace, choice of rotation method (3 pass and emmt)
# I deleted imrotate3jl because it cannot pass the adjoint test
"""
    assign_A(A, x)
    assign key values in SparseInterpolator A that are calculated from x
"""
function assign_A!(A::SparseInterpolator,
                   x::AbstractVector)
        # x was created in increasing order
        dec = ceil(Int, x[1]) - x[1]
        ncoeff = length(A.C)
        ncol = length(x)
        for i = 1:ncoeff
            if isodd(i)
                A.C[i] = dec
            else
                A.C[i] = 1 - dec
            end
        end
        if x[end] <= ncol
            A.J[end] = ceil(Int, x[end])
            for i = ncoeff-1:-1:1
                A.J[i] = max(1, A.J[end] - ceil(Int, (ncoeff - i) / 2))
            end
        else
            A.J[1] = floor(Int, x[1])
            for i = 2:ncoeff
                A.J[i] = min(ncol, A.J[1] + ceil(Int, (i - 1) / 2))
            end
        end
end

"""
    rotate_x!(output, img, tan_θ, xi, yi, A, vec_x, c_y)
    Rotate an image along x axis in clockwise direction using 1d linear interpolation,
    storing results in `output`
"""
function rotate_x!(output, img, tan_θ, xi, yi, A, vec_x, c_y)
    # rotate_x(xin, yin) = xin + (yin - c_y) * tan_θ
    for i = 1:length(yi)
        # note for future refinement: the rotate_x. step is allocating
        # vec_x .= rotate_x.(xi, yin, θ)
        vec_x .= yi[i]
        broadcast!(-, vec_x, vec_x, c_y)
        broadcast!(*, vec_x, vec_x, tan_θ)
        broadcast!(+, vec_x, vec_x, xi)
        assign_A!(A, vec_x)
        mul!((@view output[:, i]), A, (@view img[:, i])) # need mul! to avoid allocating
    end
    return output
end

"""
    rotate_x_adj!(output, img, tan_θ, xi, yi, A, vec_x, c_y)
    The adjoint of rotating an image along x axis in clockwise direction using 1d linear interpolation,
    storing results in `output`
"""
function rotate_x_adj!(output, img, tan_θ, xi, yi, A, vec_x, c_y)
    # rotate_x(xin, yin, θ) = xin + (yin - (length(yi)+1)/2) * tan(θ/2)
    for i = 1:length(yi)
        vec_x .= yi[i]
        broadcast!(-, vec_x, vec_x, c_y)
        broadcast!(*, vec_x, vec_x, tan_θ)
        broadcast!(+, vec_x, vec_x, xi)
        assign_A!(A, vec_x)
        mul!((@view output[:, i]), A', (@view img[:, i]))
    end
    return output
end

"""
    rotate_y!(output, img, sin_θ, xi, yi, A, vec_y, c_x)
    Rotate an image along y axis in clockwise direction using 1d linear interpolation,
    storing results in `output`
"""
function rotate_y!(output, img, sin_θ, xi, yi, A, vec_y, c_x)
    # rotate_y(xin, yin, θ) = (xin - (length(xi)+1)/2) * (-sin(θ)) + yin
    for i = 1:length(xi)
        # vec_y .= rotate_y.(xin, yi, θ)
        vec_y .= xi[i]
        broadcast!(-, vec_y, vec_y, c_x)
        broadcast!(*, vec_y, vec_y, sin_θ)
        broadcast!(+, vec_y, vec_y, yi)
        assign_A!(A, vec_y)
        mul!((@view output[i, :]), A, (@view img[i, :]))
    end
    return output
end

"""
    rotate_y_adj!(output, img, sin_θ, xi, yi, A, vec_y, c_x)
    The adjoint of rotating an image along y axis in clockwise direction using 1d linear interpolation,
    storing results in `output`
"""
function rotate_y_adj!(output, img, sin_θ, xi, yi, A, vec_y, c_x)
    # rotate_y(xin, yin, θ) = (xin - (length(xi)+1)/2) * (-sin(θ)) + yin
    for i = 1:length(xi)
        # vec_y .= rotate_y.(xin, yi, θ)
        vec_y .= xi[i]
        broadcast!(-, vec_y, vec_y, c_x)
        broadcast!(*, vec_y, vec_y, sin_θ)
        broadcast!(+, vec_y, vec_y, yi)
        assign_A!(A, vec_y)
        mul!((@view output[i, :]), A', (@view img[i, :]))
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
    Inplace version of rotating an image by 90/180/270 degrees
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
    imrotate3!(output, tmp, img, θ, M, N, pad_x, pad_y, A_x, A_y, vec_x, vec_y)
    Rotate an image by angle θ (must be ranging from 0 to 2π) in clockwise direction
    using a series of 1d linear interpolation
"""
function imrotate3!(output, tmp, img, θ, M, N, pad_x, pad_y, A_x, A_y, vec_x, vec_y)
    m = mod(floor(Int, 0.5 + θ/(π/2)), 4)
    mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
    tan_mod_theta = tan(mod_theta / 2)
    sin_mod_theta = - sin(mod_theta)
    # pad_x = Int((xi[end] - M) / 2)
    # pad_y = Int((yi[end] - N) / 2)
    xi = 1 : M + 2 * pad_x
    yi = 1 : N + 2 * pad_y
    c_x = (length(xi)+1)/2
    c_y = (length(yi)+1)/2
    copyto!(tmp, OffsetArrays.no_offset_view(BorderArray(img, Fill(0, (pad_x, pad_y)))))
    rot_f90!(output, tmp, m)
    rotate_x!(tmp, output, tan_mod_theta, xi, yi, A_x, vec_x, c_y)
    rotate_y!(output, tmp, sin_mod_theta, xi, yi, A_y, vec_y, c_x)
    rotate_x!(tmp, output, tan_mod_theta, xi, yi, A_x, vec_x, c_y)
    return @view tmp[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
end

"""
    imrotate3_adj!(output, tmp, img, θ, M, N, pad_x, pad_y, A_x, A_y, vec_x, vec_y)
    The adjoint of rotating an image by angle θ (must be ranging from 0 to 2π) in clockwise direction
    using a series of 1d linear interpolation
"""
function imrotate3_adj!(output, tmp, img, θ, M, N, pad_x, pad_y, A_x, A_y, vec_x, vec_y)
    m = mod(floor(Int, 0.5 + θ/(π/2)), 4)
    mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
    tan_mod_theta = tan(mod_theta / 2)
    sin_mod_theta = - sin(mod_theta)
    # pad_x = Int((xi[end] - M) / 2)
    # pad_y = Int((yi[end] - N) / 2)
    xi = 1 : M + 2 * pad_x
    yi = 1 : N + 2 * pad_y
    c_x = (length(xi)+1)/2
    c_y = (length(yi)+1)/2
    copyto!(tmp, OffsetArrays.no_offset_view(BorderArray(img, Fill(0, (pad_x, pad_y)))))
    rotate_x_adj!(output, tmp, mod_theta, xi, yi, A_x, vec_x, c_y)
    rotate_y_adj!(tmp, output, mod_theta, xi, yi, A_y, vec_y, c_x)
    rotate_x_adj!(output, tmp, mod_theta, xi, yi, A_x, vec_x, c_y)
    rot_f90_adj!(tmp, output, m) # must be two different arguments
    return @view tmp[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
end


"""
    imrotate3!(output, tmp, img, θ, M, N, pad_x, pad_y)
    Rotate an image by angle θ in clockwise direction using 2d linear interpolation
    Source code is here: https://github.com/emmt/LinearInterpolators.jl
"""
function imrotate3!(output, tmp, img, θ, M, N, pad_x, pad_y)
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
        copyto!(tmp, OffsetArrays.no_offset_view(BorderArray(img, Fill(0, (pad_x, pad_y)))))
        mul!(output, A, tmp)
        return @view output[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
    end
end

"""
    imrotate3_adj!(output, tmp, img, θ, M, N, pad_x, pad_y)
    The adjoint of rotating an image by angle θ in clockwise direction using 2d linear interpolation
    Source code is here: https://github.com/emmt/LinearInterpolators.jl
"""
function imrotate3_adj!(output, tmp, img, θ, M, N, pad_x, pad_y)
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
        copyto!(tmp, OffsetArrays.no_offset_view(BorderArray(img, Fill(0, (pad_x, pad_y)))))
        mul!(output, A', tmp)
        return @view output[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
    end
end
