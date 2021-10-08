# backproject.jl

using Main.SPECTrecon: SPECTplan, Workarray
using Main.SPECTrecon: backproject!
using BenchmarkTools: @btime
using MATLAB

function call_SPECTbackproj_matlab(mpath, views, mumap, psfs, dy)

    mat"""
    addpath($mpath)
    SPECTbackproj_matlab($views, $mumap, $psfs, $dy);
    """
end

function backproject_time()
    T = Float32
    nx = 64
    ny = 64
    nz = 40
    nview = 60

    mumap = rand(T, nx, ny, nz)

    nx_psf = 19
    nz_psf = 19
    psfs = rand(T, nx_psf, nz_psf, ny, nview)
    psfs = psfs .+ mapslices(reverse, psfs, dims = [1, 2])
    psfs = psfs ./ mapslices(sum, psfs, dims = [1, 2])

    xtrue = rand(T, nx, ny, nz)

    dy = T(4.7952)

    plan1d = SPECTplan(mumap, psfs, dy; interpidx = 1)
    plan2d = SPECTplan(mumap, psfs, dy; interpidx = 2)

    workarray1d = Vector{Workarray}(undef, plan1d.ncore)
    workarray2d = Vector{Workarray}(undef, plan2d.ncore)

    for i = 1:plan1d.ncore
        workarray1d[i] = Workarray(plan1d.T, plan1d.imgsize, plan1d.pad_fft, plan1d.pad_rot) # allocate
    end

    for i = 1:plan2d.ncore
        workarray2d[i] = Workarray(plan2d.T, plan2d.imgsize, plan2d.pad_fft, plan2d.pad_rot) # allocate
    end


    image1d = zeros(T, nx, ny, nz)
    image2d = zeros(T, nx, ny, nz)
    proj = rand(T, nx, nz, nview)

    println("backproject-1d")
    @btime backproject!($image1d, $proj, $plan1d, $workarray1d) # 277.131 ms (101482 allocations: 3.88 MiB)
    println("backproject-2d")
    @btime backproject!($image2d, $proj, $plan2d, $workarray2d) # 171.320 ms (101517 allocations: 3.88 MiB)
    mpath = pwd()
    println("backproject-matlab")
    println("Warning: Check if MIRT is installed")
    call_SPECTbackproj_matlab(mpath, proj, mumap, psfs, dy) # 194.107 ms, about 0.01 GiB
    nothing
end

# run all functions, time may vary on different machines, but should be all zero allocation.
backproject_time()
