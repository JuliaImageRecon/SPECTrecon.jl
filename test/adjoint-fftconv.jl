# adjoint-fftconv.jl
# test adjoint consistency for FFT convolution methods on very small case

using SPECTrecon: plan_psf
using SPECTrecon: fft_conv!, fft_conv_adj!, fft_conv_adj2!
using SPECTrecon: fft_conv, fft_conv_adj
using LinearMapsAA: LinearMapAA
using Test: @test, @testset, @inferred


@testset "fftconv" begin
    img = randn(Float32, 12, 8)
    ker = rand(Float64, 7, 7)
    ker = ker .+ reverse(ker, dims=:)
    ker /= sum(ker)
    out = @inferred fft_conv(img, ker)
    @test eltype(out) == Float64
end


@testset "fftconv" begin
    plan = plan_psf(10, 10, 5)
    show(stdout, "text/plain", plan)
    show(stdout, "text/plain", plan[1])
    @test sizeof(plan) isa Int
end


@testset "fftconv3" begin
    nx = 12
    nz = 10
    nx_psf = 5
    T = Float32
    plan = plan_psf(nx, nz, nx_psf; T)
    image3 = rand(T, nx, nx, nz)
    ker3 = ones(T, nx_psf, nx_psf, nx) / (nx_psf)^2
    result = similar(image3)
    fft_conv!(result, image3, ker3, plan)
    @test maximum(result) ≤ 1
    fft_conv_adj!(result, image3, ker3, plan)
    @test maximum(result) ≤ 1.5 # boundary is the sum of replicate padding
    fft_conv_adj2!(result, image3[:, 3, :], ker3, plan)
    @test maximum(result) ≤ 1.5 # boundary is the sum of replicate padding
end


@testset "adjoint-fftconv" begin
    M = 20
    N = 14
    T = Float32
    for i = 1:4 # test with different kernels
        img = randn(T, M, N)
        ker = rand(T, 5, 5)
        ker = ker .+ reverse(ker, dims=:)
        ker /= sum(ker)

        idim = (M, N)
        odim = (M, N)
        forw = img -> fft_conv(img, ker)
        back = img -> fft_conv_adj(img, ker)

        A = LinearMapAA(forw, back, (prod(odim), prod(idim)); T, odim, idim)
        @test Matrix(A') ≈ Matrix(A)' # todo
    end
end
