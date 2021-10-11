# backproject.jl

using SPECTrecon: SPECTplan
using SPECTrecon: backproject!
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
    psfs = psfs .+ mapslices(transpose, psfs, dims = [1, 2])
    psfs = psfs ./ mapslices(sum, psfs, dims = [1, 2])

    xtrue = rand(T, nx, ny, nz)

    dy = T(4.7952)

    plan1d = SPECTplan(mumap, psfs, dy; interpmeth = :one)
    plan2d = SPECTplan(mumap, psfs, dy; interpmeth = :two)


    image1d = zeros(T, nx, ny, nz)
    image2d = zeros(T, nx, ny, nz)
    proj = rand(T, nx, nz, nview)

    println("backproject-1d")
    @btime backproject!($image1d, $proj, $plan1d) # 373.614 ms (26002 allocations: 2.00 MiB)
    println("backproject-2d")
    @btime backproject!($image2d, $proj, $plan2d) # 197.220 ms (25962 allocations: 1.37 MiB)
    mpath = pwd()
    println("backproject-matlab")
    println("Warning: Check if MIRT is installed")
    call_SPECTbackproj_matlab(mpath, proj, mumap, psfs, dy) # 236.958 ms, about 0.01 GiB
    nothing
end

# run all functions, time may vary on different machines, but should be all zero allocation.
backproject_time()
