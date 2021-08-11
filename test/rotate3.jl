# using Main.SPECTrecon:imrotate3!, imrotate3_adj!, imrotate3emmt!, imrotate3emmt_adj!
using Test: @test, @testset, @test_throws, @inferred
using OffsetArrays
using ImageFiltering
using LazyAlgebra
using BenchmarkTools
@testset "imrotate3" begin
    M = 16
    N = 16
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    vec_x = zeros(Float32, M + 2 * pad_x)
    vec_y = zeros(Float32, N + 2 * pad_y)
    A_x = SparseInterpolator(LinearSpline(Float32), vec_x, length(vec_x))
    A_y = SparseInterpolator(LinearSpline(Float32), vec_y, length(vec_y))
    x = randn(Float32, M, N)
    y = randn(Float32, M, N)
    output_x = OffsetArrays.no_offset_view(padarray(x, Fill(0, (pad_x, pad_y))))
    output_y = OffsetArrays.no_offset_view(padarray(y, Fill(0, (pad_x, pad_y))))
    tmp_x = similar(output_x)
    tmp_y = similar(output_y)
    θ_list = [π/7, 3π/7, 5π/7, π, 9π/7, 11π/7, 13π/7]
    for θ in θ_list
        @test isapprox(vdot(y, imrotate3!(output_x, tmp_x, x, θ, M, N, pad_x, pad_y, A_x, A_y, vec_x, vec_y)),
                    vdot(x, imrotate3_adj!(output_y, tmp_y, y, θ, M, N, pad_x, pad_y, A_x, A_y, vec_x, vec_y)))
        @test isapprox(vdot(y, imrotate3!(output_x, tmp_x, x, θ, M, N, pad_x, pad_y)),
                    vdot(x, imrotate3_adj!(output_y, tmp_y, y, θ, M, N, pad_x, pad_y)))
    end
end

# M = 16
# N = 16
# pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
# pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
# xi = 1 : M + 2 * pad_x
# yi = 1 : N + 2 * pad_y
# vec_x = zeros(Float32, M + 2 * pad_x)
# vec_y = zeros(Float32, N + 2 * pad_y)
# A_x = SparseInterpolator(LinearSpline(Float32), vec_x, length(vec_x))
# A_y = SparseInterpolator(LinearSpline(Float32), vec_y, length(vec_y))
# x = randn(Float32, M, N)
# y = randn(Float32, M, N)
# output_x = OffsetArrays.no_offset_view(padarray(x, Fill(0, (pad_x, pad_y))))
# output_y = OffsetArrays.no_offset_view(padarray(y, Fill(0, (pad_x, pad_y))))
# tmp_x = similar(output_x)
# tmp_y = similar(output_y)
# θ_list = [π/7, 3π/7, 5π/7, π, 9π/7, 11π/7, 13π/7]
# @btime imrotate3_adj!(output_x, tmp_x, x, θ_list[1], M, N, pad_x, pad_y, A_x, A_y, vec_x, vec_y)
# A_x = SparseInterpolator(LinearSpline(Float32), vec_x, length(xi))
# A_y = SparseInterpolator(LinearSpline(Float32), vec_y, length(yi))
# imrotate3!(output_x, tmp_x, x, θ_list[2], M, N, pad_x,
#                 pad_y, A_x, A_y, vec_x, vec_y)
# imrotate3emmt!(output_y, tmp_y, x, θ_list[2], M, N, pad_x, pad_y)
#
#
# @btime rotate_x!(output_x, tmp_x, θ_list[1], xi, yi, A, vec_x)
# rotate_x_adj!(output_y, tmp_x, -θ_list[1], xi, yi, vec_x)
