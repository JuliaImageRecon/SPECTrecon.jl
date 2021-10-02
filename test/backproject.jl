# backproject.jl

using Main.SPECTrecon: backproject, backproject!
using MAT
using BenchmarkTools
using Test: @test, @testset, detect_ambiguities


@testset "backproject" begin

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

    image1d = zeros(T, nx, ny, nz)
    image2d = zeros(T, nx, ny, nz)

    @btime backproject!($image1d, $proj_jeff, $plan1d, $workarray1d)
    @btime backproject!($image2d, $proj_jeff, $plan2d, $workarray2d)
    # running on a remote server using vscode
    # 1d interp 20.357 s (416171 allocations: 15.41 MiB)
    # 2d interp 5.220 s (416276 allocations: 15.41 MiB)

    # running on a remote server using atom
end
