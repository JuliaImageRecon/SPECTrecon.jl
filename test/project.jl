using Main.SPECTrecon:project, backproject
using Test: @test, @testset, @test_throws, @inferred
using LinearAlgebra
using LinearMapsAA
using BenchmarkTools
@testset "project" begin
    nx = 16
    ny = 16
    nz = 16
    nview = 16
    mumap = convert(Array{Float32, 3}, 0.01 * rand(Float32, nx, ny, nz))
    psfs = convert(Array{Float32, 4}, 1/25 * ones(Float32, 5, 5, ny, nview))
    dy = Float32(4.7952)

    # 1d linear interpolation
    A = LinearMapAA(x -> SPECTrecon.project(x, mumap, psfs, nview, dy; interpidx = 1),
                    y -> SPECTrecon.backproject(y, mumap, psfs, nview, dy; interpidx = 1),
                    (nx*nz*nview, nx*ny*nz);
                    T=Float32, idim = (nx, ny, nz), odim = (nx, nz, nview))

    # 2d bilinear interpolation
    B = LinearMapAA(x -> SPECTrecon.project(x, mumap, psfs, nview, dy; interpidx = 2),
                    y -> SPECTrecon.backproject(y, mumap, psfs, nview, dy; interpidx = 2),
                    (nx*nz*nview, nx*ny*nz);
                    T=Float32, idim = (nx, ny, nz), odim = (nx, nz, nview))

    x = randn(Float32, nx, ny, nz)
    y = randn(Float32, nx, nz, nview)
    @test isapprox(vdot(y, A * x), vdot(x, A' * y))
    @test isapprox(vdot(y, B * x), vdot(x, B' * y))
end

nx = 16
ny = 16
nz = 16
nview = 16
mumap = convert(Array{Float32, 3}, 0.01 * rand(Float32, nx, ny, nz))
psfs = convert(Array{Float32, 4}, 1/25 * ones(Float32, 5, 5, ny, nview))
dy = Float32(4.7952)
plan = SPECTrecon.SPECTplan(mumap, psfs, nview, dy; interpidx = 1)
workarray = Vector{SPECTrecon.Workarray_s}(undef, plan.ncore)
for i = 1:plan.ncore
    workarray[i] = SPECTrecon.Workarray_s(plan) # allocate
end
x = randn(Float32, nx, ny, nz)
y = randn(Float32, nx, nz, nview)
views = zeros(Float32, nx, nz, nview)
@btime SPECTrecon.project!(views, plan, workarray, x)
# 1d interp, 34.702 ms (203927 allocations: 10.00 MiB)


plan = SPECTrecon.SPECTplan(mumap, psfs, nview, dy; interpidx = 2)
workarray = Vector{SPECTrecon.Workarray_s}(undef, plan.ncore)
for i = 1:plan.ncore
    workarray[i] = SPECTrecon.Workarray_s(plan) # allocate
end
x = randn(Float32, nx, ny, nz)
y = randn(Float32, nx, nz, nview)
views = zeros(Float32, nx, nz, nview)
@btime SPECTrecon.project!(views, plan, workarray, x)
# 2d interp, 33.247 ms (76446 allocations: 4.30 MiB)


img = randn(Float32, plan.nx, plan.nz)
@btime SPECTrecon.my_conv!(workarray[1].padimg,
                  img,
                  (@view plan.psfs[:, :, 8, 8]),
                  workarray[1].img_compl,
                  workarray[1].ker_compl,
                  workarray[1].tmp_compl,
                  plan)
# 90.450 μs (133 allocations: 9.89 KiB)
