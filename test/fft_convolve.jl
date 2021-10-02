# fft_convolve.jl

using Main.SPECTrecon: imfilter3!
using Main.SPECTrecon: fft_conv!, fft_conv_adj!
using LazyAlgebra:vdot
using FFTW: plan_fft!, plan_ifft!
using ImageFiltering: centered, imfilter
using ImageFiltering: imfilter, centered
using MIRTjim: jim
using BenchmarkTools
using Test: @test, @testset, detect_ambiguities


@testset "imfilter3!" begin

    N = 64
    T = Float32
    #img = zeros(T, N, N); img[20:50, 20:40] .= ones(31,21)
    img = rand(T, N, N) # random image that goes all the way to the edge!
    output = similar(img)
    ker = rand(T, 3, 3) / 9
    img_compl = similar(img, Complex{T})
    ker_compl = similar(img_compl)
    fft_plan = plan_fft!(img_compl)
    ifft_plan = plan_ifft!(img_compl)
    copyto!(img_compl, img)
    imfilter3!(output, img_compl, reverse(ker), ker_compl, fft_plan, ifft_plan)
    y = imfilter(img, centered(ker), "circular")
    @test isapprox(y, output)
    # jim(jim(output, "my"), jim(y, "julia"), jim(output - y, "diff"), jim(img), gui=true)
    @btime imfilter3!($output, $img_compl, $ker, $ker_compl, $fft_plan, $ifft_plan)
    # 29.746 μs (0 allocations: 0 bytes)

end


@testset "fft_conv!" begin

    M = 200
    N = 64
    T = Float32
    img = zeros(T, M, N)
    img[20:150, 20:40] .= rand(131,21)
    output = similar(img)
    ker = ones(T, 3, 3) / 9
    fftpadsize = (28, 28, 32, 32)

    img_compl = zeros(Complex{T}, 256, 128)
    workmat = zeros(T, 256, 128)
    ker_compl = similar(img_compl)
    fft_plan = plan_fft!(img_compl)
    ifft_plan = plan_ifft!(img_compl)
    fft_conv!(output, workmat, img, ker, fftpadsize,
            img_compl, ker_compl, fft_plan, ifft_plan)
    # todo: test output vs normal fft version
    # no need to test this function, as we already tested imfilter3
    @btime fft_conv!($output, $workmat, $img, $ker, $fftpadsize,
            $img_compl, $ker_compl, $fft_plan, $ifft_plan)
    # 455.066 μs (0 allocations: 0 bytes)

end


@testset "fft_conv_adj!" begin

    M = 200
    N = 64
    T = Float32
    img = zeros(T, M, N)
    img[20:150, 20:40] .= rand(131,21)
    output = similar(img)
    ker = ones(T, 3, 3) / 9
    fftpadsize = (28, 28, 32, 32)

    img_compl = zeros(Complex{T}, 256, 128)
    workmat = zeros(T, 256, 128)
    workvec1 = zeros(T, 128)
    workvec2 = zeros(T, 256)
    ker_compl = similar(img_compl)
    fft_plan = plan_fft!(img_compl)
    ifft_plan = plan_ifft!(img_compl)
    fft_conv_adj!(output, workmat, workvec1, workvec2, img, ker, fftpadsize,
            img_compl, ker_compl, fft_plan, ifft_plan)
    # todo: test correctness, see test folder
    @btime fft_conv_adj!($output, $workmat, $workvec1, $workvec2, $img, $ker,
        $fftpadsize, $img_compl, $ker_compl, $fft_plan, $ifft_plan)
    # 449.281 μs (0 allocations: 0 bytes)

end


@testset "adj-test" begin
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
        fft_conv!(output_x, workmat, x, ker, fftpadsize,
                                img_compl, ker_compl, fft_plan, ifft_plan)

        fft_conv_adj!(output_y, workmat, workvec1, workvec2, y, kerev, fftpadsize,
                                    img_compl, ker_compl, fft_plan, ifft_plan)
        @test isapprox(vdot(y, output_x), vdot(x, output_y))
    end
end
