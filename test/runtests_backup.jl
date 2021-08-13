using Main.SPECTrecon
using Test: @testset
using MAT
using Plots:plot
using MIRTjim
using LinearMapsAA
using LinearAlgebra
path = "/Users/zongyuli/SPECTrecon.jl/test/"
file = matopen(path*"mumap208.mat")
mumap = read(file, "mumap208")
close(file)
# mumap = 0.01 * ones(Float32, 128, 128, 80)

file = matopen(path*"psf_208.mat")
psfs = read(file, "psf_208")
close(file)
# psf = zeros(Float32,37,37,128,128)
# psf[19,19,:,:] .= 1

file = matopen(path*"xtrue.mat")
xtrue = convert(Array{Float32, 3}, read(file, "xtrue"))
close(file)
# mumap = 0.01 * xtrue
file = matopen(path*"proj_jeff_newmumap.mat")
proj_jeff = read(file, "proj_jeff")
close(file)
nx, ny, nz = size(xtrue)
nview = size(psfs, 4)
dy = Float32(4.7952)
plan = SPECTrecon.SPECTplan(mumap, psfs, nview, dy; interpidx = 2)
workarray = Vector{SPECTrecon.Workarray_s}(undef, plan.ncore)
for i = 1:plan.ncore
    workarray[i] = SPECTrecon.Workarray_s(plan) # allocate
end
# SPECTplan(mumap, psfs, nview, dy; interpidx = 1)
# views = project(xtrue, mumap, psf, nview, dy)
views = zeros(Float32, nx, nz, nview)
@btime SPECTrecon.project!(views, plan, workarray, xtrue)
# 3.773 s (2765314 allocations: 788.88 MiB)
# @allocated project(plan, xtrue)
@btime SPECTrecon.backproject!(views, plan, workarray, xtrue)
# 3.148 s (1872602 allocations: 771.74 MiB)
# currently code
# A = LinearMapAA(x -> project(plan, x), y -> backproject(plan, y), (nx*nz*nview, nx*ny*nz);
#                 T=Float32, idim = (nx, ny, nz), odim = (nx, nz, nview))
#
# x = rand(Float32, nx, ny, nz)
# y = rand(Float32, nx, nz, nview)
# isapprox(vdot(A * x, y), vdot(x, A' * y))

# (2.0443462e7 - 2.0453376e7) / 2.0453376e7

# isapprox(vec(y)'*vec(A * x), vec(A' * y)'*vec(x))

nor = x -> x / maximum(x)
idx = 40
nrmse(x, xtrue) = norm(vec(x - xtrue)) / norm(vec(xtrue))
# plot(jim(proj_ellipse[:,:,idx], "Jeff"), jim(views[:,:,idx], "My"),
        # jim(nor(proj_ellipse[:,:,idx]), "Jeff normalized"), jim(nor(views[:,:,idx]), "My normalized"))
# savefig(path*"ellipse_jeff_vs_my_idx"*string(idx)*".pdf")
#
# plot(jim(abs.(nor(proj_ellipse[:,:,idx]) - nor(view[:,:,idx])), "abs diff"))
# savefig(path*"ellipse_diff_jeff_vs_my_idx"*string(idx)*".pdf")

plot(jim(proj_jeff[:,:,idx], "Jeff"), jim(views[:,:,idx], "My"),
        jim(nor(proj_jeff[:,:,idx]), "Jeff normalized"), jim(nor(views[:,:,idx]), "My normalized"))
# x
# savefig(path*"proj_jeff_vs_my_idx"*string(idx)*".pdf")

# plot(jim(proj_jeff[:,:,idx] - views[:,:,idx], "diff for proj"))
# plot(jim(nor(proj_jeff[:,:,idx]) - nor(views[:,:,idx]), "diff for nor proj"))
# savefig(path*"proj_diff_jeff_vs_my_idx"*string(idx)*".pdf")
# plot(jim(abs.(proj_jeff[:,:,idx] - view[:,:,idx]), "abs diff for proj"))
# plot(jim(abs.(nor(proj_jeff[:,:,idx]) - nor(view[:,:,idx])), "abs diff for normalized proj"))
# savefig(path*"petvp6_diff_jeff_vs_my_idx"*string(idx)*".pdf")
e3 = zeros(128)
for idx = 1:128
    e3[idx] = nrmse(views[:,:,idx], proj_jeff[:,:,idx])
end
plot((0:127)/128*360, e3 * 100, xticks = 0:45:360, xlabel = "degree", ylabel = "NRMSE (%)", label = "")

# savefig(path*"nrmse_myproj_vs_jeff.pdf")
# nrmse(views[:,:,idx], proj_jeff[:,:,idx])
# nrmse(nor(view[:,:,idx]), nor(proj_jeff[:,:,idx]))
nrmse(views, proj_jeff)
x
# nrmse(nor(view), nor(proj_jeff))
# nrmse before 0.5 slice thickness: 3.21%
# nrmse after ....................: 0.40%, caused by imrotate
# nrmse after ....................: 0.23%, caused by my_rotate
# plot(jim(abs.(nor(proj_jeff[:,1:end-5,idx]) - nor(view[:,1:end-5,idx])), "abs diff"))
# savefig(path*"petvp6_diff_end-5_jeff_vs_my_idx"*string(idx)*".pdf")
# plot(proj_jeff[:,end,idx],label = "Jeff")
# plot!(view[:,end,idx], label = "My")
# title!("Bottom line profile")
# savefig(path*"petvp6_bottom_jeff_vs_my_idx"*string(idx)*".pdf")
# @testset "SPECTrecon.jl" begin
#     # todo
# end
# using Images, CoordinateTransformations, Rotations, TestImages, OffsetArrays
# img = zeros(128, 128)
# img[40:80, 40:80] .= rand(41, 41)
# # tfm = recenter(RotMatrix(-π/4), center(img))
# tfm = LinearMap(RotMatrix(-π/6))
#
# imgw = warp(centered(img), tfm, axes(centered(img)), 0)
#
# θ = 21 * π/128
# A = LinearMapAA(x -> vec(warp(centered(reshape(x, M, N)), LinearMap(RotMatrix(θ)), axes(centered(reshape(x, M, N))), degree = BSpline(Cubic(Line(OnGrid()))), 0)),
#                 y -> vec(warp(centered(reshape(y, M, N)), LinearMap(RotMatrix(-θ)), axes(centered(reshape(y, M, N))), degree = Linear(), 0)),
#                 (M*N, M*N); T = Float32)
# B = LinearMapAA(x -> vec(imrotate(reshape(x, M, N), θ, axes(reshape(x, M, N)), 0)),
#                 y -> vec(imrotate(reshape(y, M, N), -θ, axes(reshape(y, M, N)), 0)),
#                 (M*N, M*N); T=Float32)
# jim(imgw)
# x = zeros(128, 128)
# x[30:90, 30:90] .= ones(61,61)
# y = zeros(128, 128)
# y[30:90, 30:90] .= ones(61, 61)
# isapprox(y'*(A*x), (A'*y)'*x)
# isapprox(y'*(B*x), (B'*y)'*x)
# y'*(A*x) - (A'*y)'*x
# y'*(B*x) - (B'*y)'*x
# rotate = x -> OffsetArrays.no_offset_view(warp(centered(x),
#                     LinearMap(RotMatrix(- θ)),
#                     axes(centered(x)), degree = Linear(), 0))
# rotate(img)



# image = rand(Float32, 128,128,80)
# mumap = zeros(Float32, 128,128,80)
# psfs = zeros(Float32,37,37,128,128)
# psfs[19,19,:,:] .= 1
# nview = 1
# dy = Float32(4.7952)
# plan = SPECTplan(mumap, psfs, nview, dy)
# views = zeros(promote_type(eltype(image), Float32), plan.nx, plan.nz, plan.nview)
# project!(views, plan, image)
# x
# using LinearMapsAA
# θ = π/6
# M = 129
# N = 129
# A = LinearMapAA(x -> vec(imrotate(reshape(x, M, N), -θ, axes(reshape(x, M, N)), 0)),
#                 y -> vec(imrotate(reshape(y, M, N), θ, axes(reshape(y, M, N)), 0)),
#                 (M*N, M*N); T=Float32)
# # # isapprox(Matrix(A'), Matrix(A)')
# x = zeros(M, N)
# y = zeros(M, N)
# x[45:85,25:85] .= rand(41, 61)
# y[45:85,25:85] .= rand(41, 61)
# x = vec(x)
# y = vec(y)
# isapprox(y'*(A*x), (A'*y)'*x)
# abs(y'*(A*x) - (A'*y)'*x) / abs(y'*(A*x))


# itp = interpolate(rand(16,16), BSpline(Linear()); dims = 2)
#
# u = rand(5, 5, 5)
# t = 0:4
# itpx = x -> extrapolate(interpolate(x, (BSpline(Linear()))), 0)
# itpy = x -> extrapolate(interpolate(x, (NoInterp(), BSpline(Linear()))), 0)
# img = rand(128, 128, 80)
# for i = 1:80
#     itp = itpy(img[:,:,i])
#
#     # rotate along y
#     # interpolate along x
#     # rotate along x
# end
# using Rotations
# using CoordinateTransformations
# img = rand(128, 128)
# tfm = img -> recenter(RotMatrix(-π/4), center(img))
# warp(img, tfm, axes(img), 0)
# A = LinearMapAA(x -> vec(warp(img, tfm, axes(img), 0)),
#                 y -> vec(imrotate(reshape(y, M, N), -θ, axes(reshape(y, M, N)), 0)),
#                 (M*N, M*N); T=Float32)


# function rotate_y(img, θ)
#     M, N = size(img) # M and N should be odd
#     xi = 1:M
#     yi = 1:N
#     ic = extrapolate(interpolate((xi, yi), img, (NoInterp(), Gridded(Linear()))), 0)
#     # ic = extrapolate(interpolate((xi, yi), img, (NoInterp(), BSpline(Linear()))), 0)
#     # ic = LinearInterpolation((xi, yi), img, extrapolation_bc = 0)
#     # rotate_y(xin, yin, θ) = - xin * tan(θ) + yin / cos(θ)
#     # rotate_x(xin, yin, θ) = xin * cos(θ) + yin * sin(θ)
#     rotate_y(xin, yin, θ) = - (xin - (M-1)/2) * tan(θ) + (yin - (N-1)/2) / cos(θ)
#     # return [ic(xin, rotate_y(xin, yin, θ)...) for xin in xi, yin in reverse(yi)]
#     return [ic(xin, rotate_y(xin, yin, θ) + (N-1)/2) for xin in xi, yin in yi]
#     # return [ic(xin, rotate_y(xin, yin, θ)) for xin in xi, yin in yi]
# end
#
# function rotate_x(img, θ)
#     M, N = size(img) # M and N should be odd
#     xi = 1:M
#     yi = 1:N
#     ic = extrapolate(interpolate((xi, yi), img, (Gridded(Linear()), NoInterp())), 0)
#     # ic = extrapolate(interpolate((xi, yi), img, (NoInterp(), BSpline(Linear()))), 0)
#     # ic = LinearInterpolation((xi, yi), img, extrapolation_bc = 0)
#     # rotate_y(xin, yin, θ) = - xin * tan(θ) + yin / cos(θ)
#     # rotate_x(xin, yin, θ) = xin * cos(θ) + yin * sin(θ)
#     rotate_x(xin, yin, θ) = (xin - (M-1)/2) * cos(θ) + (yin - (N-1)/2) * sin(θ)
#     # return [ic(xin, rotate_y(xin, yin, θ)...) for xin in xi, yin in reverse(yi)]
#     return [ic(rotate_x(xin, yin, θ) + (M-1)/2, yin) for xin in xi, yin in yi]
#     # return [ic(xin, rotate_y(xin, yin, θ)) for xin in xi, yin in yi]
# end

# θ = π/6


# function rotate_x2(img, θ)
#     M, N = size(img) # M and N should be odd
#     xi = 1:M
#     yi = 1:N
#     # ic = extrapolate(interpolate((xi, yi), img, (Gridded(Linear()), NoInterp())), 0)
#     rotate_x(xin, yin, θ) = (xin - (M+1)/2) * cos(θ) + (yin - (N+1)/2) * sin(θ) + (M+1)/2
#     tmp = zeros(eltype(img), M, N)
#     for yin in yi
#         ic = LinearInterpolation(xi, img[:, yin], extrapolation_bc = 0)
#         tmp[:, yin] .= ic.(rotate_x.(xi, yin, θ))
#     end
#     return tmp
#     # return [ic(rotate_x(xin, yin, θ) + (M-1)/2, yin) for xin in xi, yin in yi]
# end
#
# function rotate_y2(img, θ)
#     M, N = size(img) # M and N should be odd
#     xi = 1:M
#     yi = 1:N
#     # ic = extrapolate(interpolate((xi, yi), img, (NoInterp(), Gridded(Linear()))), 0)
#     rotate_y(xin, yin, θ) = (xin - (M+1)/2) * (-tan(θ)) + (yin - (N+1)/2) / cos(θ) + (N+1)/2
#     tmp = zeros(eltype(img), M, N)
#     for xin in xi
#         ic = LinearInterpolation(yi, img[xin, :], extrapolation_bc = 0)
#         tmp[xin, :] .= ic.(rotate_y.(xin, yi, θ))
#     end
#     return tmp
# end
#
# using LinearMapsAA
# θ = 33/128*(2π)
# M = 10
# N = 10
# x = zeros(M, N)
# y = zeros(M, N)
# x[4:7,4:7] .= rand(4,4)
# y[4:7,4:7] .= rand(4,4)
# # x[45:85,25:85] .= ones(41, 61)
# # y[45:85,25:85] .= ones(41, 61)
#
# function rotate_x3(img, θ)
#     M, N = size(img) # M and N should be odd
#     xi = 1:M
#     yi = 1:N
#     # xi = -(M-1)/2 : (M-1)/2
#     # yi = -(N-1)/2 : (N-1)/2
#     # ic = extrapolate(interpolate((xi, yi), img, (Gridded(Linear()), NoInterp())), 0)
#     rotate_x(xin, yin, θ) = (xin - (M+1)/2) + (yin - (N+1)/2) * tan(θ/2) + (M+1)/2
#     tmp = zeros(eltype(img), M, N)
#     for (i, yin) in enumerate(yi)
#         ic = LinearInterpolation(xi, img[:, i], extrapolation_bc = 0)
#         tmp[:, i] .= ic.(rotate_x.(xi, yin, θ))
#     end
#     return tmp
#     # return [ic(rotate_x(xin, yin, θ) + (M-1)/2, yin) for xin in xi, yin in yi]
# end
#
# function rotate_y3(img, θ)
#     M, N = size(img) # M and N should be odd
#     xi = 1:M
#     yi = 1:N
#     # xi = -(M-1)/2 : (M-1)/2
#     # yi = -(N-1)/2 : (N-1)/2
#     # ic = extrapolate(interpolate((xi, yi), img, (NoInterp(), Gridded(Linear()))), 0)
#     rotate_y(xin, yin, θ) = (xin - (M+1)/2) * (-sin(θ)) + (yin - (N+1)/2) + (N+1)/2
#     tmp = zeros(eltype(img), M, N)
#     for (i, xin) in enumerate(xi)
#         # ic = extrapolate(interpolate(yi, img[xin, :]), 0)
#         ic = LinearInterpolation(yi, img[i, :], extrapolation_bc = 0)
#         tmp[i, :] .= ic.(rotate_y.(xin, yi, θ))
#     end
#     return tmp
# end
# function rot_back(img, m)
#     if m == 0
#         return img
#     elseif m == 1
#         return rotl90(img)
#     elseif m == 2
#         return rot180(img)
#     elseif m == 3
#         return rotr90(img)
#     else
#         throw("invalid m!")
#     end
# end
# function my_rotate_test(img, θ)
#     M, N = size(img)
#     m = mod(floor(Int, (θ + π/4) / (π/2)), 4)
#     mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
#     pad_x = ceil(Int, 1 + M * sqrt(2)/2)
#     pad_y = ceil(Int, 1 + N * sqrt(2)/2)
#     return rot_back(rotate_x3(rotate_y3(rotate_x3(OffsetArrays.no_offset_view(padarray(img, Pad(:reflect, pad_x, pad_y))),
#                 mod_theta), mod_theta), mod_theta), m)[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N]
# end
# todo: θ = π

# A = LinearMapAA(x -> imrotate(x, -θ, axes(x), 0),
#                 y -> imrotate(y, θ, axes(y), 0),
#                 (M*N, M*N); idim = (M, N), odim = (M, N), T=Float64)
# B = LinearMapAA(x -> rotate_x3(rotate_y3(rotate_x3(x, θ), θ), θ),
#                 y -> rotate_x3(rotate_y3(rotate_x3(y, -θ), -θ), -θ),
#                 (M*N, M*N); idim = (M, N), odim = (M, N), T = Float64)
# B = LinearMapAA(x -> my_rotate_test(x, θ),
#                 y -> my_rotate_test(y, -θ),
#                 (M*N, M*N); idim = (M, N), odim = (M, N), T = Float64)
# tform(x) = recenter(RotMatrix{2}(-θ), center(reshape(x, M, N)))
# tform_adj(x) = recenter(RotMatrix{2}(θ), center(reshape(x, M, N)))
# D = LinearMapAA(x -> vec(warp(reshape(x, M, N), tform(x), axes(reshape(x, M, N)), 0)),
#                 y -> vec(warp(reshape(y, M, N), tform_adj(y), axes(reshape(x, M, N)), 0)),
#                 (M*N, M*N); T= Float64)
# isapprox(vec(y)'*vec(A*x), vec(A'*y)'*vec(x))
# isapprox(vec(y)'*vec(B*x), vec(B'*y)'*vec(x))
# isapprox(y'*(C*x), (C'*y)'*x)
# isapprox(y'*(D*x), (D'*y)'*x)
# abs(y'*(B*x) - (B'*y)'*x) / abs(y'*(B*x))
# jim(A*x - B*x, "diff")
# jim(A * x)
# plot(jim(x), jim(A*x), jim(A*x .- x), title = "imrotate")
# plot(jim(x), jim(B*x), jim(B*x .- x), title = "my rotate")
# norm(vec(A*x) - vec(B*x)) / norm(vec(A*x))
# jim(reshape(A*x, M, N) - reverse(reshape(x, M, N)))
# jim(reshape(A*x, M, N) - reshape(B * x, M, N))
# jim(reshape(A*x, M, N) - reshape(C * x, M, N))
# jim(reshape(A*x, M, N) - reshape(D * x, M, N))
# C = LinearMapAA(x -> imrotate(x, -θ, axes(x), 0),
#                 y -> imrotate(y, θ, axes(y), 0),
#                 (M*N, M*N); idim = (M, N), odim = (M, N), T=Float32)
# plot(jim(reshape(A*x, M, N) - reshape(x, M, N), "imrotate"),
#     jim(reshape(B*x, M, N) - reshape(x, M, N), "my rotate"))
# jim(C * img)
# step 1: idim and odim
# step 2: debug my rotate
# x

# θ = π/6
# img = zeros(128, 128)
# img[40:80, 40:80] .= rand(41, 41)
# imgr = my_rotate(img, θ)
# imgr_j = imrotate(img, -θ, axes(img), 0)
# plot(jim(imgr), jim(imgr_j))
# jim(imgr - imgr_j)
# norm(vec(imgr) - vec(imgr_j)) / norm(vec(imgr_j))
#
# imgr_adj = rotate_y_adj(img, θ)
# jim(imgr_adj)
# M, N = size(img)
# A = LinearMapAA(x -> vec(rotate_y(reshape(x, M, N), θ)),
#                 y -> vec(rotate_y_adj(reshape(y, M, N), θ)),
#                 (M*N, M*N); T = Float64)
# jim(reshape(A' * vec(img), M, N))
#
# x = zeros(128, 128)
# x[30:90, 30:90] .= rand(61,61)
# x = vec(x)
# y = zeros(128, 128)
# y[30:90, 30:90] .= 2 * ones(61, 61)
# y = vec(y)
#
# isapprox(y'*(A*x), (A'*y)'*x)
#
# B = LinearMapAA(x -> vec(imrotate(reshape(x, M, N), θ, axes(reshape(x, M, N)), 0)),
#                 y -> vec(imrotate(reshape(y, M, N), -θ, axes(reshape(y, M, N)), 0)),
#                 (M*N, M*N); T=Float32)
# isapprox(y'*(B*x), (B'*y)'*x)
# isapprox(imrotate(img, π, axes(img), 0), reverse(img))
# jim(reverse(img))
# jim(imrotate(img, π, axes(img), 0) - reverse(img))
# imgr = rotate_x(rotate_y(rotate_x(img, θ), θ), θ)
# jim(imgr)
# plot(jim(imgr), jim(imrotate(img, -θ, axes(img), 0)))
# plot(jim(imgr - imrotate(img, -θ, axes(img), 0)))

# bar(;kwargs...) = kwargs.data.a
#
# function filler!(x, i)
#    x[:,:] .+= i
# end
#
# function filler!(x)
#     for i = 1:size(x,3)
#         filler!((@view x[:,:,i]), i)
#     end
#     return x
# end
#
# y = zeros(2,3,4)
# filler!(y)
# A = rand(128, 128)
# B = padarray(A, Pad(:replicate,(0,0),(1,1)))
