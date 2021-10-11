# project.jl

using SPECTrecon: SPECTplan
using SPECTrecon: project!
using MAT
using LinearAlgebra: norm


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

    plan1d = SPECTplan(mumap, psfs, dy; interpmeth = :one)
    plan2d = SPECTplan(mumap, psfs, dy; interpmeth = :two)


    (nx, ny, nz) = size(xtrue)
    nview = size(psfs, 4)

    views1d = zeros(T, nx, nz, nview)
    views2d = zeros(T, nx, nz, nview)

    project!(views1d, xtrue, plan1d)
    project!(views2d, xtrue, plan2d)

    nrmse(x, xtrue) = norm(vec(x - xtrue)) / norm(vec(xtrue))

    err1d = zeros(nview)
    for idx = 1:nview
        err1d[idx] = nrmse(views1d[:,:,idx], proj_jeff[:,:,idx])
    end

    err2d = zeros(nview)
    for idx = 1:nview
        err2d[idx] = nrmse(views2d[:,:,idx], proj_jeff[:,:,idx])
    end

    return err1d, err2d
    # 1d interp: 2e-7 nrmse
    # 2d interp: 4e-3 nrmse
end

# shows accuracy
err1d, err2d = project_error()
nrmse1d = sum(err1d) / length(err1d)
nrmse2d = sum(err2d) / length(err2d)
@show nrmse1d
@show nrmse2d
nothing
# plot(nrmse1d)
