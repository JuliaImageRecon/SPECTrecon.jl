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


    dy = T(4.7952)

    for interpmeth in (:one, :two)
        for mode in (:fast, :mem)
            plan = SPECTplan(mumap, psfs, dy; interpmeth, mode)
            xtrue = rand(T, nx, ny, nz)
            views = zeros(T, nx, nz, nview)
            println(string(interpmeth)*", "*string(mode))
            @btime project!($views, $xtrue, $plan)
        end
    end

    mpath = pwd()
    xtrue = rand(T, nx, ny, nz)
    println("project-matlab")
    println("Warning: Check if MIRT is installed")
    call_SPECTproj_matlab(mpath, xtrue, mumap, psfs, dy) # 216.518 ms, about 0.01 GiB
    nothing
end

# run all functions, time may vary on different machines, will allocate ~4MB memory.
project_time()

#= one, fast
    378.348 ms (25983 allocations: 1.37 MiB)
one, mem
    793.918 ms (31257 allocations: 1.76 MiB)
two, fast
    289.804 ms (25982 allocations: 1.37 MiB)
two, mem
    618.369 ms (31268 allocations: 1.76 MiB)
MIRT
    230.125 ms
=#
