# SPECTplan.jl

using Main.SPECTrecon: SPECTplan, Workarray
using BenchmarkTools: @btime
using MATLAB

function call_SPECTplan_matlab(mpath, mumap, psfs, dy)

    mat"""
    addpath($mpath)
    SPECTplan_matlab($mumap, $psfs, $dy);
    """
end


function SPECTplan_time()
    T = Float32
    nx = 64
    ny = 64
    nz = 40
    nview = 60

    mumap = rand(T, nx, ny, nz)

    nx_psf = 19
    nz_psf = 19
    dy = T(4.7952)

    psfs = rand(T, nx_psf, nz_psf, ny, nview)
    psfs = psfs .+ mapslices(reverse, psfs, dims = [1, 2])
    psfs = psfs ./ mapslices(sum, psfs, dims = [1, 2])

    plan = SPECTplan(mumap, psfs, dy)
    workarray = Vector{Workarray}(undef, plan.ncore)
    for i = 1:plan.ncore
        workarray[i] = Workarray(plan.T, plan.imgsize, plan.pad_fft, plan.pad_rot) # allocate
    end
    println("SPECTplan")
    @btime plan = SPECTplan($mumap, $psfs, $dy)
    # 7.016 ms (50093 allocations: 9.16 MiB)
    println("Workarrays")
    @btime workarray = Vector{Workarray}(undef, $plan.ncore)
    # 75.968 ns (1 allocation: 1008 bytes)

    @btime for i = 1:$plan.ncore
        $workarray[i] = Workarray($plan.T, $plan.imgsize, $plan.pad_fft, $plan.pad_rot) # allocate
    end
    # 1.376 ms (2432 allocations: 2.00 MiB)
    mpath = pwd()
    println("SPECTplan_matlab")
    println("Warning: Check if MIRT is installed")
    call_SPECTplan_matlab(mpath, mumap, psfs, dy)
    # 0.941699 seconds, real memory size: 1.30 GB
    nothing
end


# run all functions, time may vary on different machines, will alllocate ~100 MB memory.
SPECTplan_time()
