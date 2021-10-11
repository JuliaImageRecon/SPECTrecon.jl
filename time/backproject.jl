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

    dy = T(4.7952)

    for interpmeth in (:one, :two)
        for mode in (:fast, :mem)
            plan = SPECTplan(mumap, psfs, dy; interpmeth, mode)
            image = zeros(T, nx, ny, nz)
            proj = rand(T, nx, nz, nview)
            println(string(interpmeth)*", "*string(mode))
            @btime backproject!($image, $proj, $plan)
        end
    end

    mpath = pwd()
    proj = rand(T, nx, nz, nview)
    println("backproject-matlab")
    println("Warning: Check if MIRT is installed")
    call_SPECTbackproj_matlab(mpath, proj, mumap, psfs, dy) # 236.958 ms, about 0.01 GiB
    nothing
end

# run all functions, time may vary on different machines, but should be all zero allocation.
backproject_time()
#=one, fast
  343.419 ms (26022 allocations: 2.00 MiB)
one, mem
  379.044 ms (33691 allocations: 1.96 MiB)
two, fast
  208.163 ms (25981 allocations: 1.37 MiB)
two, mem
  246.847 ms (33666 allocations: 1.96 MiB)
MIRT
  212.261 ms
=#
