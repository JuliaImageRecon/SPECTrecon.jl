# rotatez.jl

using SPECTrecon: rotl90!, rotr90!, rot180!, rot_f90!, rot_f90_adj!
using Test: @test, @testset, @test_throws


@testset "rotl90!" begin
    T = Float32
    N = 20
    A = rand(T, N, N)
    B = similar(A, N, N)
    rotl90!(B, A)
    @test isequal(B, rotl90(A))
end


@testset "rotr90!" begin
    T = Float32
    N = 20
    A = rand(T, N, N)
    B = rand(T, N, N)
    rotr90!(B, A)
    @test isequal(B, rotr90(A))
end


@testset "rot180!" begin
    T = Float32
    N = 20
    A = rand(T, N, N)
    B = rand(T, N, N)
    rot180!(B, A)
    @test isequal(B, rot180(A))
end


@testset "rot_f90!" begin
    T = Float32
    N = 20
    A = rand(T, N, N)
    B = rand(T, N, N)
    @test_throws String rot_f90!(A, B, -1)
    @test_throws String rot_f90_adj!(A, B, -1)
    @test_throws String rot_f90!(A, B, 4)
    @test_throws String rot_f90_adj!(A, B, 4)
end
