using Main.SPECTrecon
using Test: @test, @testset, @test_throws, @inferred
using LazyAlgebra
using InterpolationKernels
using LinearInterpolators
using BenchmarkTools
using Plots:plot
using MIRTjim:jim
using FFTW

@testset "fft_convolve" begin
    M = 200
    N = 64
    T = Float32
    testnum = 100
    fftpadsize = (28, 28, 32, 32)
    img_compl = zeros(Complex{T}, 256, 128)
    ker_compl = similar(img_compl)
    workmat = zeros(T, 256, 128)
    workvec1 = zeros(T, 128)
    workvec2 = zeros(T, 256)
    fft_plan = plan_fft!(img_compl)
    ifft_plan = plan_ifft!(img_compl)
    for t = 1:testnum
        x = randn(T, M, N)
        output_x = similar(x)
        y = randn(T, M, N)
        output_y = similar(y)
        ker = rand(T, 5, 5)
        ker /= sum(ker)
        kerev = reverse(ker)
        Main.SPECTrecon.fft_conv!(output_x, workmat, x, ker, fftpadsize,
                                img_compl, ker_compl, fft_plan, ifft_plan)

        Main.SPECTrecon.fft_conv_adj!(output_y, workmat, workvec1, workvec2, y, kerev, fftpadsize,
                                    img_compl, ker_compl, fft_plan, ifft_plan)
        @test isapprox(vdot(y, output_x), vdot(x, output_y))
    end
end
