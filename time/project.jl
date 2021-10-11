# project.jl

using BenchmarkTools: @btime
using SPECTrecon: SPECTplan
using SPECTrecon: project!
using MATLAB

function call_SPECTproj_matlab(mpath, image, mumap, psfs, dy)

    mat"""
    addpath($mpath)
    SPECTproj_matlab($image, $mumap, $psfs, $dy);
    """
end

function project_time()
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


    views1d = zeros(T, nx, nz, nview)
    views2d = zeros(T, nx, nz, nview)

    println("project-1d")
    @btime project!($views1d, $xtrue, $plan1d) # 394.127 ms (25961 allocations: 1.37 MiB)
    println("project-2d")
    @btime project!($views2d, $xtrue, $plan2d) # 275.048 ms (25962 allocations: 1.37 MiB)
    mpath = pwd()
    println("project-matlab")
    println("Warning: Check if MIRT is installed")
    call_SPECTproj_matlab(mpath, xtrue, mumap, psfs, dy) # 216.518 ms, about 0.01 GiB
    nothing
end

# run all functions, time may vary on different machines, will allocate ~4MB memory.
project_time()
