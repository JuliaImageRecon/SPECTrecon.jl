using Main.SPECTrecon:imrotate3!, imrotate3_adj!
using Test: @test, @testset, @test_throws, @inferred
using OffsetArrays
using ImageFiltering
using LazyAlgebra
using InterpolationKernels
using LinearInterpolators
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
        @test isapprox(vdot(y, SPECTrecon.imrotate3!(output_x, tmp_x, x, θ, M, N, pad_x, pad_y, A_x, A_y, vec_x, vec_y)),
                    vdot(x, SPECTrecon.imrotate3_adj!(output_y, tmp_y, y, θ, M, N, pad_x, pad_y, A_x, A_y, vec_x, vec_y)))
        @test isapprox(vdot(y, SPECTrecon.imrotate3!(output_x, tmp_x, x, θ, M, N, pad_x, pad_y)),
                    vdot(x, SPECTrecon.imrotate3_adj!(output_y, tmp_y, y, θ, M, N, pad_x, pad_y)))
    end
end

M = 16
N = 16
pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
xi = 1 : M + 2 * pad_x
yi = 1 : N + 2 * pad_y
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
@btime SPECTrecon.imrotate3!(output_x, tmp_x, x, θ_list[1], M, N, pad_x, pad_y, A_x, A_y, vec_x, vec_y)
# 1d interp, 24.366 μs (239 allocations: 11.19 KiB)
@btime SPECTrecon.imrotate3!(output_x, tmp_x, x, θ_list[1], M, N, pad_x, pad_y)
# 2d interp, 6.121 μs (2 allocations: 80 bytes)
@btime SPECTrecon.imrotate3_adj!(output_y, tmp_y, y, θ_list[1], M, N, pad_x, pad_y, A_x, A_y, vec_x, vec_y)
# 1d interp, 27.178 μs (329 allocations: 15.12 KiB)
@btime SPECTrecon.imrotate3_adj!(output_x, tmp_x, x, θ_list[1], M, N, pad_x, pad_y)
# 2d interp, 6.298 μs (2 allocations: 80 bytes)
@btime SPECTrecon.assign_A!(A_x, vec_x)
# 129.127 ns (0 allocations: 0 bytes)
