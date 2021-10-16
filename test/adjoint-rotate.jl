# adjoint-rotate.jl
# test adjoint consistency for rotate methods on very small case

using SPECTrecon: plan_rotate
using SPECTrecon: imrotate, imrotate_adj
using SPECTrecon: imrotate!, imrotate_adj!
using LinearMapsAA: LinearMapAA
using Test: @test, @testset


@testset "rotate" begin
    plan = plan_rotate(10)
    show(isinteractive() ? stdout : devnull, "text/plain", plan)
    show(isinteractive() ? stdout : devnull, "text/plain", plan[1])
    @test sizeof(plan) isa Int
    @test sizeof(plan[1]) isa Int
end


@testset "rotate3" begin
    for method in (:one, :two)
        nx = 20
        θ = π/6
        T = Float32
        plan = plan_rotate(nx; T, method)
        image3 = rand(T, nx, nx, 4)
        result1 = similar(image3)
        result2 = similar(result1)
        imrotate!(result1, image3, θ, plan)
        @test maximum(result1) ≤ 1.01
        imrotate!(result2, image3, θ, plan, :thread)
        @test result1 == result2

        imrotate_adj!(result1, image3, θ, plan)
        @test maximum(result1) ≤ 1.5
        imrotate_adj!(result2, image3, θ, plan, :thread)
        @test result1 == result2
    end
end


@testset "adjoint-rotate" begin
    Ntest = 7
    θ_list = (0:Ntest-1) / Ntest * 2π
    M = 16
    N = M
    T = Float32

    for method in (:one, :two), θ in θ_list
        idim = (M, N)
        odim = (M, N)
        x = randn(T, M, N)
        forw = x -> imrotate(x, θ; method)
        back = x -> imrotate_adj(x, θ; method)
        A = LinearMapAA(forw, back, (prod(odim), prod(idim)); T, odim, idim)
        @test Matrix(A') ≈ Matrix(A)'
    end
end
