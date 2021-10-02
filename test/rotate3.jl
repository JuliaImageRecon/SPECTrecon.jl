# rotate3.jl

using Main.SPECTrecon: linearinterp!, rotate_x!, rotate_y!
using Main.SPECTrecon: rotate_x_adj!, rotate_y_adj!
using Main.SPECTrecon: rotl90!, rotr90!, rot180!
using Main.SPECTrecon: imrotate3!, imrotate3_adj!
using LazyAlgebra:vdot
using LinearInterpolators: SparseInterpolator, LinearSpline
using MIRTjim:jim
using BenchmarkTools
using ImageFiltering: padarray
using ImageTransformations: imrotate
using MIRTjim: jim
using Interpolations: Linear
using OffsetArrays
using Test: @test, @testset, detect_ambiguities


@testset "linearinterp!" begin

    T = Float32
    x = rand() * ones(T, 100)
    y = copy(x)
    interp_x = SparseInterpolator(LinearSpline(T), x, length(x))
    interp_y = SparseInterpolator(LinearSpline(T), y, length(y))
    linearinterp!(interp_x, y)
    @test isequal(interp_x.C, interp_y.C)
    @test isequal(interp_x.J, interp_y.J)
    @btime linearinterp!($interp_x, $x) # 421.226 ns (0 allocations: 0 bytes)

end


@testset "rotate_x!" begin

    N = 100 # assume M = N
    T = Float32
    img = rand(T, N, N)
    output = rand(T, N, N)
    θ = T(3*π/11)
    xi = T.(1:N)
    yi = T.(1:N)
    interp_x = SparseInterpolator(LinearSpline(T), xi, length(xi))
    workvec = rand(T, N)
    c_y = 1
    # todo: test values, how to test values?
    @btime rotate_x!($output, $img, $θ, $xi, $yi, $interp_x, $workvec, $c_y)
    # 69.822 μs (0 allocations: 0 bytes)

end


@testset "rotate_x_adj!" begin

    N = 100 # assume M = N
    T = Float32
    img = rand(T, N, N)
    output = rand(T, N, N)
    θ = T(3*π/11)
    xi = T.(1:N)
    yi = T.(1:N)
    interp_x = SparseInterpolator(LinearSpline(T), xi, length(xi))
    workvec = rand(T, N)
    c_y = 1
    @btime rotate_x_adj!($output, $img, $θ, $xi, $yi, $interp_x, $workvec, $c_y)
    # 84.405 μs (0 allocations: 0 bytes)

end


@testset "rotate_y!" begin

    N = 100 # assume M = N
    T = Float32
    img = rand(T, N, N)
    output = rand(T, N, N)
    θ = T(3*π/11)
    xi = T.(1:N)
    yi = T.(1:N)
    interp_x = SparseInterpolator(LinearSpline(T), xi, length(xi))
    workvec = rand(T, N)
    c_x = 1
    @btime rotate_y!($output, $img, $θ, $xi, $yi, $interp_x, $workvec, $c_x)
    # 72.389 μs (0 allocations: 0 bytes)

end


@testset "rotate_y_adj!" begin

    N = 100 # assume M = N
    T = Float32
    img = rand(T, N, N)
    output = rand(T, N, N)
    θ = T(3*π/11)
    xi = T.(1:N)
    yi = T.(1:N)
    interp_x = SparseInterpolator(LinearSpline(T), xi, length(xi))
    workvec = rand(T, N)
    c_x = 1
    @btime rotate_y_adj!($output, $img, $θ, $xi, $yi, $interp_x, $workvec, $c_x)
    # 89.847 μs (0 allocations: 0 bytes)

end


@testset "rotl90!" begin

    T = Float32
    N = 100
    A = rand(T, N, N)
    B = rand(T, N, N)
    rotl90!(B, A)
    @test isequal(B, rotl90(A))
    @btime rotl90!($B, $A) # 3.608 μs (0 allocations: 0 bytes)

end


@testset "rotr90!" begin

    T = Float32
    N = 100
    A = rand(T, N, N)
    B = rand(T, N, N)
    rotr90!(B, A)
    @test isequal(B, rotr90(A))
    @btime rotr90!($B, $A) # 3.700 μs (0 allocations: 0 bytes)

end


@testset "rot180!" begin

    T = Float32
    N = 100
    A = rand(T, N, N)
    B = rand(T, N, N)
    rot180!(B, A)
    @test isequal(B, rot180(A))
    @btime rot180!($B, $A) # 3.798 μs (0 allocations: 0 bytes)

end

"""
`imrotate3-1d` visualization
"""
function imrotate3_1d_vis()
    T = Float32
    M = 100
    N = 100
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    workvec_x = zeros(T, M + 2 * pad_x)
    workvec_y = zeros(T, N + 2 * pad_y)
    A_x = SparseInterpolator(LinearSpline(T), workvec_x, length(workvec_x))
    A_y = SparseInterpolator(LinearSpline(T), workvec_y, length(workvec_y))
    img = zeros(T, M, N)
    img[30:50, 20:60] .= 1
    output = similar(img)
    workmat1 = OffsetArrays.no_offset_view(padarray(img, Fill(0, (pad_x, pad_y))))
    workmat2 = similar(workmat1)
    θ = 3*π/16
    imrotate3!(output, workmat1, workmat2, img, θ, A_x, A_y, workvec_x, workvec_y)
    output2 = imrotate(img, -θ, axes(img), method = Linear(), fill = 0)
    @btime imrotate3!($output, $workmat1, $workmat2, $img, $θ, $A_x, $A_y, $workvec_x, $workvec_y)
    # 557.079 μs (0 allocations: 0 bytes)
    return output, output2
end

imo_1d, imj_1d = imrotate3_1d_vis()
jim(jim(imo_1d, "my"), jim(imj_1d, "julia"), jim(imo_1d - imj_1d, "diff"))

"""
`imrotate3_adj-1d` visualization
"""
function imrotate3_1d_adj_vis()
    T = Float32
    M = 100
    N = 100
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    workvec_x = zeros(T, M + 2 * pad_x)
    workvec_y = zeros(T, N + 2 * pad_y)
    A_x = SparseInterpolator(LinearSpline(T), workvec_x, length(workvec_x))
    A_y = SparseInterpolator(LinearSpline(T), workvec_y, length(workvec_y))
    img = zeros(T, M, N)
    img[30:50, 20:60] .= 1
    output = similar(img)
    workmat1 = OffsetArrays.no_offset_view(padarray(img, Fill(0, (pad_x, pad_y))))
    workmat2 = similar(workmat1)
    θ = 3π/16
    imrotate3_adj!(output, workmat1, workmat2, img, θ, A_x, A_y, workvec_x, workvec_y)
    output2 = imrotate(img, θ, axes(img), method = Linear(), fill = 0)
    @btime imrotate3_adj!($output, $workmat1, $workmat2, $img, $θ, $A_x, $A_y, $workvec_x, $workvec_y)
    # 544.053 μs (0 allocations: 0 bytes)
    return output, output2
end

imo_1d_adj, imj_1d_adj = imrotate3_1d_adj_vis()
jim(jim(imo_1d_adj, "my"), jim(imj_1d_adj, "julia"), jim(imo_1d_adj - imj_1d_adj, "diff"))


function imrotate3_2d_vis()

    T = Float32
    M = 100
    N = 100
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    img = zeros(T, M, N)
    img[30:60,20:70] .= 1
    output = similar(img)
    workmat1 = OffsetArrays.no_offset_view(padarray(img, Fill(0, (pad_x, pad_y))))
    workmat2 = similar(workmat1)
    θ = 3π/16
    imrotate3!(output, workmat1, workmat2, img, θ)
    output2 = imrotate(img, -θ, axes(img), method = Linear(), fill = 0)
    @btime imrotate3!($output, $workmat1, $workmat2, $img, $θ)
    # 220.708 μs (0 allocations: 0 bytes)
    return output, output2

end

imo_2d, imj_2d = imrotate3_2d_vis()
jim(jim(imo_2d, "my"), jim(imj_2d, "julia"), jim(imo_2d - imj_2d, "diff"))


function imrotate3_2d_adj_vis()

    T = Float32
    M = 100
    N = 100
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    img = zeros(T, M, N)
    img[30:60,20:70] .= 1
    output = similar(img)
    workmat1 = OffsetArrays.no_offset_view(padarray(img, Fill(0, (pad_x, pad_y))))
    workmat2 = similar(workmat1)
    θ = 3π/16
    imrotate3_adj!(output, workmat1, workmat2, img, θ)
    output2 = imrotate(img, θ, axes(img), method = Linear(), fill = 0)
    @btime imrotate3_adj!($output, $workmat1, $workmat2, $img, $θ)
    # 216.833 μs (0 allocations: 0 bytes)
    return output, output2

end

imo_2d_adj, imj_2d_adj = imrotate3_2d_adj_vis()
jim(jim(imo_2d_adj, "my"), jim(imj_2d_adj, "julia"), jim(imo_2d_adj - imj_2d_adj, "diff"))
# I see a lot of artifacts! But suprisingly, it can pass the adjoint test ???

@testset "adjtest-1d" begin
    Ntest = 256
    θ_list = (0:Ntest-1) / Ntest * 2π
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
        imrotate3!(output_x, workmat1_x, workmat2_x, x, θ, A_x, A_y, workvec_x, workvec_y)
        imrotate3_adj!(output_y, workmat1_y, workmat2_y, y, θ, A_x, A_y, workvec_x, workvec_y)
        @test isapprox(vdot(y, output_x), vdot(x, output_y))
    end
end


@testset "adjtest-2d" begin
    Ntest = 256
    θ_list = (0:Ntest-1) / Ntest * 2π
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
        imrotate3!(output_x, workmat1_x, workmat2_x, x, θ)
        imrotate3_adj!(output_y, workmat1_y, workmat2_y, y, θ)
        @test isapprox(vdot(y, output_x), vdot(x, output_y))
    end
end
