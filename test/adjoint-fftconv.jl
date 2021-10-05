# adjoint-fftconv.jl
# test adjoint consistency for FFT convolution methods on very small case

using SPECTrecon: fft_conv!, fft_conv_adj!, Power2
using FFTW: plan_fft!, plan_ifft!
using LinearMapsAA: LinearMapAA
using Test: @test, @testset


@testset "adjoint-fftconv" begin
    M = 200 # todo: make smaller
#   M = 20
    N = 64
    T = Float32
    fftpadsize = (28, 28, 32, 32)
    img_compl = zeros(Complex{T}, 256, 128)
    ker_compl = similar(img_compl)
    workmat = zeros(T, 256, 128)
    workvec1 = zeros(T, 128)
    workvec2 = zeros(T, 256)
    fft_plan = plan_fft!(img_compl)
    ifft_plan = plan_ifft!(img_compl)

#   x = randn(T, M, N)
#   output_x = similar(x)
#   y = randn(T, M, N)
#   output_y = similar(y)
    ker = rand(T, 11, 11) # change from 5 -> 11
    ker /= sum(ker)
    kerev = reverse(ker)

    forw!(y,x) = fft_conv!(y, workmat, x,
        ker, fftpadsize, img_compl, ker_compl, fft_plan, ifft_plan)

    back!(x,y) = fft_conv_adj!(x, workmat, workvec1, workvec2, y,
        kerev, fftpadsize, img_compl, ker_compl, fft_plan, ifft_plan)

    idim = (M,N)
    odim = (M,N)
    A = LinearMapAA(forw!, back!, (prod(odim), prod(idim)); T, odim, idim)
#   @test Matrix(A') ≈ Matrix(A)' # todo
end
