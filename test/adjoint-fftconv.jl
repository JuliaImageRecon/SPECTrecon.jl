# adjoint-fftconv.jl
# test adjoint consistency for FFT convolution methods on very small case

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
