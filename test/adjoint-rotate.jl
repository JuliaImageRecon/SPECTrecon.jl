# adjoint-rotate.jl
# test adjoint consistency for rotate methods on very small case

using SPECTrecon: linearinterp!, rotate_x!, rotate_y!
using SPECTrecon: rotate_x_adj!, rotate_y_adj!
using SPECTrecon: rotl90!, rotr90!, rot180!
using SPECTrecon: imrotate3!, imrotate3_adj!
using LinearAlgebra: dot
using LinearInterpolators: SparseInterpolator, LinearSpline
using LinearMapsAA: LinearMapAA
using Test: @test, @testset
using Random: seed!


@testset "adjoint-rotate" begin
    Ntest = 7
    θ_list = (0:Ntest-1) / Ntest * 2π
    M,N = 16, 16 # todo: test non-square?
    T = Float32
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    workvec_x = zeros(T, M + 2 * pad_x)
    workvec_y = zeros(T, N + 2 * pad_y)
    workmat2_x = Array{T}(undef, M + 2 * pad_x, N + 2 * pad_y)
    workmat2_y = Array{T}(undef, M + 2 * pad_x, N + 2 * pad_y)
    workmat1_x = Array{T}(undef, M + 2 * pad_x, N + 2 * pad_y)
    workmat1_y = Array{T}(undef, M + 2 * pad_x, N + 2 * pad_y)
    A_x = SparseInterpolator(LinearSpline(T), workvec_x, length(workvec_x))
    A_y = SparseInterpolator(LinearSpline(T), workvec_y, length(workvec_y))

    for θ in θ_list
		idim = (M,N)
		odim = (M,N)

        forw1!(y, x) = imrotate3!(y, workmat1_x, workmat2_x, x, θ)
        back1!(x, y) = imrotate3_adj!(x, workmat1_y, workmat2_y, y, θ)
		A = LinearMapAA(forw1!, back1!, (prod(odim), prod(idim)); T, odim, idim)
        @test Matrix(A') ≈ Matrix(A)' # 3-pass 1D version

        forw2!(y, x) = imrotate3!(y, workmat1_x, workmat2_x, x, θ, A_x, A_y, workvec_x, workvec_y)
        back2!(x, y) = imrotate3_adj!(x, workmat1_y, workmat2_y, y, θ, A_x, A_y, workvec_x, workvec_y)
		A = LinearMapAA(forw2!, back2!, (prod(odim), prod(idim)); T, odim, idim)
        @test Matrix(A') ≈ Matrix(A)' # 2D version
    end
end
