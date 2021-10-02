# project.jl

using Main.SPECTrecon: SPECTplan, Workarray
using Main.SPECTrecon: project, project!
using Main.SPECTrecon: backproject, backproject!
using MAT
using LazyAlgebra: vdot
using BenchmarkTools
using LinearAlgebra: norm
using Test: @test, @testset, detect_ambiguities


@testset "project" begin

    T = Float32
    path = "./data/"
    file = matopen(path*"mumap208.mat")
    mumap = read(file, "mumap208")
    close(file)

    file = matopen(path*"psf_208.mat")
    psfs = read(file, "psf_208")
    close(file)

    file = matopen(path*"xtrue.mat")
    xtrue = convert(Array{Float32, 3}, read(file, "xtrue"))
    close(file)

    file = matopen(path*"proj_jeff_newmumap.mat")
    proj_jeff = read(file, "proj_jeff")
    close(file)

    dy = T(4.7952)
    nview = size(psfs, 4)

    plan1d = SPECTplan(mumap, psfs, nview, dy; interpidx = 1)
    plan2d = SPECTplan(mumap, psfs, nview, dy; interpidx = 2)

    workarray1d = Vector{Workarray}(undef, plan1d.ncore)
    workarray2d = Vector{Workarray}(undef, plan2d.ncore)

    for i = 1:plan1d.ncore
        workarray1d[i] = Workarray(plan1d.T, plan1d.imgsize, plan1d.pad_fft, plan1d.pad_rot) # allocate
    end

    for i = 1:plan2d.ncore
        workarray2d[i] = Workarray(plan2d.T, plan2d.imgsize, plan2d.pad_fft, plan2d.pad_rot) # allocate
    end


    (nx, ny, nz) = size(xtrue)
    nviews = size(psfs, 4)

    views1d = zeros(T, nx, nz, nviews)
    views2d = zeros(T, nx, nz, nviews)

    project!(views1d, xtrue, plan1d, workarray1d)
    project!(views2d, xtrue, plan2d, workarray2d)

    nrmse(x, xtrue) = norm(vec(x - xtrue)) / norm(vec(xtrue))

    err1d = zeros(nviews)
    for idx = 1:nviews
        err1d[idx] = nrmse(views1d[:,:,idx], proj_jeff[:,:,idx])
    end

    err2d = zeros(nview)
    for idx = 1:nviews
        err2d[idx] = nrmse(views2d[:,:,idx], proj_jeff[:,:,idx])
    end

    @show sum(err1d) / length(err1d) * 100
    @show sum(err2d) / length(err2d) * 100
    # 1d interp: 1e-5% nrmse
    # 2d interp: 0.382% nrmse
    @btime project!($views1d, $xtrue, $plan1d, $workarray1d)
    @btime project!($views2d, $xtrue, $plan2d, $workarray2d)
    # running on a remote server using vscode:
    # 1d interp 20.481 s (541597 allocations: 18.64 MiB)
    # 2d interp 5.205 s (541644 allocations: 18.64 MiB)

    # running on a remote server using atom
end


@testset "adj-test" begin
    T = Float32
    path = "./data/"
    file = matopen(path*"mumap208.mat")
    mumap = read(file, "mumap208")
    close(file)

    file = matopen(path*"psf_208.mat")
    psfs = read(file, "psf_208")
    close(file)
    dy = T(4.80)

    (nx, ny, nz) = size(mumap)
    nviews = size(psfs, 4)

    x = randn(T, nx, ny, nz)
    y = randn(T, nx, nz, nviews)

    output_x = project(x, mumap, psfs, nviews, dy; interpidx = 1)
    output_y = backproject(y, mumap, psfs, nviews, dy; interpidx = 1)
    @test isapprox(vdot(y, output_x), vdot(x, output_y))

    output_x = project(x, mumap, psfs, nviews, dy; interpidx = 2)
    output_y = backproject(y, mumap, psfs, nviews, dy; interpidx = 2)
    @test isapprox(vdot(y, output_x), vdot(x, output_y))
end
