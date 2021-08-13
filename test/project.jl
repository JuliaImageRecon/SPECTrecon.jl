using Main.SPECTrecon:project, backproject
using Test: @test, @testset, @test_throws, @inferred
using LinearMapsAA
using LazyAlgebra
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
plan = SPECTrecon.SPECTplan(mumap, psfs, nview, dy; interpidx = 2)
workarray = Vector{SPECTrecon.Workarray_s}(undef, plan.ncore)
for i = 1:plan.ncore
    workarray[i] = SPECTrecon.Workarray_s(plan) # allocate
end
views = zeros(Float32, nx, nz, nview)
image = randn(Float32, nx, ny, nz)
# viewidx = 2
@btime SPECTrecon.project!(views, plan, workarray, image)
# 1d interp, 19.991 ms (57805 allocations: 2.81 MiB)
# 2d interp, 19.595 ms (44418 allocations: 2.48 MiB)
# only rotate, 613.328 μs (5520 allocations: 249.75 KiB)
# no my_conv!, 1.428 ms (31050 allocations: 1.01 MiB)
# no inside for-loop, 979.757 μs (13609 allocations: 527.53 KiB)
# copyto, 985.286 μs (13605 allocations: 795.41 KiB)
x = randn(1000, 1000)
y = randn(1, 1000)
@btime y .= sum((@view x[1:100, :]), dims = 1)
@btime y .= dropdims(sum((@view x[1:100, :]), dims = 1); dims = 1)
@btime y .= (@view sum((@view x[1:100, :]), dims = 1)[1, :])
@btime for i = 1:100
    # y .+= (@view x[i, :])
    broadcast!(+, y, y, (@view x[i, :]))
end

@btime y[1] = sum((@view x[1:30]))
@btime for i = 1:30
    y[1] += x[i]
end
@btime SPECTrecon.backproject!(image, plan, workarray, views)
# 1d interp, 3.149 ms (29378 allocations: 1.47 MiB)
# 2d interp, 2.462 ms (29417 allocations: 1.47 MiB)
@btime imfilter!(workarray[1].padimg,
        plan.padrepl((@view plan.imgr[:, 3, :])),
        (@view plan.psfs[:, :, 3, 3]), NoPad(), Algorithm.FFT())
