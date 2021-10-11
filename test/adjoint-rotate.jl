# adjoint-rotate.jl
# test adjoint consistency for rotate methods on very small case

using SPECTrecon: imrotate1, imrotate1_adj
using SPECTrecon: imrotate2, imrotate2_adj
using LinearMapsAA: LinearMapAA
using Test: @test, @testset


@testset "adjoint-rotate" begin
    Ntest = 128
    θ_list = (0:Ntest-1) / Ntest * 2π
    M = 16
	N = 16
    T = Float32

    for θ in θ_list
		idim = (M, N)
		odim = (M, N)
		x = randn(T, M, N)
        forw1 = x -> imrotate1(x, θ)
        back1 = x -> imrotate1_adj(x, θ)
		A = LinearMapAA(forw1, back1, (prod(odim), prod(idim)); T, odim, idim)
        @test Matrix(A') ≈ Matrix(A)' # 3-pass 1D version

        forw2 = x -> imrotate2(x, θ)
        back2 = x -> imrotate2_adj(x, θ)
		A = LinearMapAA(forw2, back2, (prod(odim), prod(idim)); T, odim, idim)
        @test Matrix(A') ≈ Matrix(A)' # 2D version
    end
end
