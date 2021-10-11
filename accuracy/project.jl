# project.jl

using SPECTrecon: SPECTplan
using SPECTrecon: project!
using MAT
using LinearAlgebra: norm

function cal_nrmse(x, xtrue)
    len = size(xtrue, 3)
    err = zeros(eltype(xtrue), len)
    for l = 1:len
        err[l] = norm(vec(x[:,:,l]) - vec(xtrue[:,:,l])) / norm(vec(xtrue[:,:,l]))
    end
    nrmse = sum(err) / len
    return nrmse
end

function project_error()
    T = Float32
    path = "../data/"
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

    (nx, ny, nz) = size(xtrue)

    for interpmeth in (:one, :two)
        for mode in (:fast, :mem)
            println(string(interpmeth)*", "*string(mode))
            plan = SPECTplan(mumap, psfs, dy; interpmeth, mode)
            views = zeros(T, nx, nz, nview)
            project!(views, xtrue, plan)
            nrmse = cal_nrmse(views, proj_jeff)
            @show nrmse
        end
    end

    # 1d interp: 2e-7 nrmse
    # 2d interp: 4e-3 nrmse
end

# shows accuracy
project_error()
nothing
# plot(nrmse1d)
