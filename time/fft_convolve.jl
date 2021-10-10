# fft_convolve.jl

using BenchmarkTools: @btime
using Main.SPECTrecon: plan_psf
using Main.SPECTrecon: fft_conv!, fft_conv_adj!


function fft_conv_time()
    M = 200
    N = 64
    T = Float32
    img = zeros(T, M, N)
    img[20:150, 20:40] .= rand(131,21)
    output = similar(img)
    nx_psf = 5
    ker = ones(T, nx_psf, nx_psf) / nx_psf^2
    plan = plan_psf(M, N, nx_psf; nthread = 1, T = T)[1]

    println("fft_conv")
    @btime fft_conv!($output, $img, $ker, $plan)
    # 565.631 μs (0 allocations: 0 bytes)
    println("fft_conv_adj")
    @btime fft_conv_adj!($output, $img, $ker, $plan)
    # 561.371 μs (0 allocations: 0 bytes)
    nothing
end


# run all functions, time may vary on different machines, but should be all zero allocation.
fft_conv_time()
