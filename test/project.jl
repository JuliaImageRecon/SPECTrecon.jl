using Main.SPECTrecon:project, backproject
using Test: @test, @testset, @test_throws, @inferred
using LinearMapsAA
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
