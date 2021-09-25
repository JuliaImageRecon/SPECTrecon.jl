using Main.SPECTrecon
using Test: @test, @testset, @test_throws, @inferred
using LazyAlgebra
using InterpolationKernels
using LinearInterpolators
using BenchmarkTools
using Plots:plot
using MIRTjim:jim
@testset "imrotate3" begin
    θ_list = rand(1000) * 2π
    M = 64
    N = 64
    T = Float32
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    workvec_x = zeros(T, M + 2 * pad_x)
    workvec_y = zeros(T, N + 2 * pad_y)
    A_x = SparseInterpolator(LinearSpline(T), workvec_x, length(workvec_x))
    A_y = SparseInterpolator(LinearSpline(T), workvec_y, length(workvec_y))
    x = randn(T, M, N)
    y = randn(T, M, N)
    output_x = zeros(T, M, N)
    output_y = zeros(T, M, N)
    workmat2_x = zeros(T, M + 2 * pad_x, N + 2 * pad_y)
    workmat2_y = zeros(T, M + 2 * pad_x, N + 2 * pad_y)
    workmat1_x = zeros(T, M + 2 * pad_x, N + 2 * pad_y)
    workmat1_y = zeros(T, M + 2 * pad_x, N + 2 * pad_y)
    for θ in θ_list
        Main.SPECTrecon.imrotate3!(output_x, workmat1_x, workmat2_x, x, θ, A_x, A_y, workvec_x, workvec_y)
        Main.SPECTrecon.imrotate3_adj!(output_y, workmat1_y, workmat2_y, y, θ, A_x, A_y, workvec_x, workvec_y)
        @test isapprox(vdot(y, output_x), vdot(x, output_y))
    end
end

@testset "imrotate3" begin
    θ_list = rand(1000) * 2π
    M = 64
    N = 64
    T = Float32
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    x = randn(T, M, N)
    y = randn(T, M, N)
    output_x = similar(x)
    output_y = similar(y)
    workmat2_x = zeros(T, M + 2 * pad_x, N + 2 * pad_y)
    workmat2_y = zeros(T, M + 2 * pad_x, N + 2 * pad_y)
    workmat1_x = zeros(T, M + 2 * pad_x, N + 2 * pad_y)
    workmat1_y = zeros(T, M + 2 * pad_x, N + 2 * pad_y)
    for θ in θ_list
        Main.SPECTrecon.imrotate3!(output_x, workmat1_x, workmat2_x, x, θ)
        Main.SPECTrecon.imrotate3_adj!(output_y, workmat1_y, workmat2_y, y, θ)
        @test isapprox(vdot(y, output_x), vdot(x, output_y))
    end
end

# visualization
M = 16
N = 16
T = Float32
pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
workvec_x = zeros(T, M + 2 * pad_x)
workvec_y = zeros(T, N + 2 * pad_y)
A_x = SparseInterpolator(LinearSpline(T), workvec_x, length(workvec_x))
A_y = SparseInterpolator(LinearSpline(T), workvec_y, length(workvec_y))
img = zeros(T, M, N)
img[4:13, 5:12] .= 1
output_1d = zeros(T, M, N)
workmat1_1d = zeros(T, M + 2 * pad_x, N + 2 * pad_y)
workmat2_1d = zeros(T, M + 2 * pad_x, N + 2 * pad_y)
output_2d = zeros(T, M, N)
workmat1_2d = zeros(T, M + 2 * pad_x, N + 2 * pad_y)
workmat2_2d = zeros(T, M + 2 * pad_x, N + 2 * pad_y)
θ = -π/7
Main.SPECTrecon.imrotate3!(output_1d, workmat1_1d, workmat2_1d, img, θ, A_x, A_y, workvec_x, workvec_y)
Main.SPECTrecon.imrotate3!(output_2d, workmat1_2d, workmat2_2d, img, θ)
plot(jim(img, "img"), jim(output_1d, "1d"), jim(output_2d, "2d"),
    jim(output_2d - output_1d, "diff"))
