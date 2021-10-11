# adjoint-rotate.jl
# test adjoint consistency for rotate methods on very small case

using SPECTrecon: imrotate1, imrotate1_adj
using SPECTrecon: imrotate2, imrotate2_adj
using SPECTrecon: plan_rotate
using SPECTrecon: imrotate!, imrotate_adj!
using LinearMapsAA: LinearMapAA
using Test: @test, @testset


@testset "rotate" begin
    plan = plan_rotate(10)
    show(stdout, "text/plain", plan)
    show(stdout, "text/plain", plan[1])
    @test sizeof(plan) isa Int
end


@testset "rotate3" begin
    for method in (:one, :two)
        nx = 20
        T = Float32
        plan = plan_rotate(nx; T, method)
        image3 = rand(T, nx, nx, 4)
        result = similar(image3)
        imrotate!(result, image3, π/6, plan)
        @test maximum(result) ≤ 1.1
        imrotate_adj!(result, image3, -π/6, plan)
        @test maximum(result) ≤ 1.1
    end
end


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
