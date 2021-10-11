# adjoint-fftconv.jl
# test adjoint consistency for FFT convolution methods on very small case

using SPECTrecon: fft_conv, fft_conv_adj
using LinearMapsAA: LinearMapAA
using Test: @test, @testset


@testset "adjoint-fftconv" begin
    M = 40
    N = 24
    T = Float32
    testnum = 20
    # test with different kernels
    for i = 1:testnum
        img = randn(T, M, N)
        ker = rand(T, 7, 7)
        ker = ker .+ reverse(ker)
        ker = ker .+ ker'
        ker /= sum(ker)

        idim = (M, N)
        odim = (M, N)
        forw = img -> fft_conv(img, ker)
        back = img -> fft_conv_adj(img, ker)

        A = LinearMapAA(forw, back, (prod(odim), prod(idim)); T, odim, idim)
        @test Matrix(A') ≈ Matrix(A)' # todo
    end
end
