# project.jl

export project, project!


"""
    project!(view, plan, image, viewidx)
SPECT projection of `image` into a single `view` with index `viewidx`.
The `view` must be pre-allocated but need not be initialized to zero.
"""
function project!(
    view::AbstractMatrix{<:RealU},
    image::AbstractArray{<:RealU, 3},
    plan::SPECTplan,
    viewidx::Int,
)

    # rotate image and mumap using multiple processors
    nz = plan.imgsize[3] # prepare to loop over slices
    spawner(plan.nthread, nz) do buffer_id, iz
        # rotate image in plan.imgr
        imrotate!(
            (@view plan.imgr[:, :, iz]),
            (@view image[:, :, iz]),
            plan.viewangle[viewidx],
            plan.planrot[buffer_id],
        )
        # rotate mumap and store in plan.mumapr
        imrotate!(
            (@view plan.mumapr[:, :, iz]),
            (@view plan.mumap[:, :, iz]),
            plan.viewangle[viewidx],
            plan.planrot[buffer_id],
        )
    end

    # apply depth-dependent attenuation and blur to each y plane
    ny = plan.imgsize[2] # prepare to loop over y planes
    spawner(plan.nthread, ny) do buffer_id, iy
        # account for half of the final slice thickness
        scale3dj!(plan.exp_mumapr[buffer_id], plan.mumapr, iy, -0.5)

        for j in 1:iy
            plus3dj!(plan.exp_mumapr[buffer_id], plan.mumapr, j)
        end

        broadcast!(*, plan.exp_mumapr[buffer_id], plan.exp_mumapr[buffer_id], - plan.dy)
        broadcast!(exp, plan.exp_mumapr[buffer_id], plan.exp_mumapr[buffer_id])

        # apply depth-dependent attenuation
        mul3dj!(plan.imgr, plan.exp_mumapr[buffer_id], iy)

        fft_conv!(
            (@view plan.add_img[:, iy, :]),
            (@view plan.imgr[:, iy, :]),
            (@view plan.psfs[:, :, iy, viewidx]),
            plan.planpsf[buffer_id],
        )
    end

    copy3dj!(view, plan.add_img, 1) # initialize accumulator
    for y in 2:plan.imgsize[2] # accumulate to get total view
        plus3dj!(view, plan.add_img, y)
    end

    # plan.add_img[2] # why does julia allocate (on heap!?) here?

    return view
end


"""
    project!(view, plan, image, thid, viewidx)
SPECT projection of `image` into a single `view` with index `viewidx`.
The `view` must be pre-allocated but need not be initialized to zero.
"""
function project!(
    view::AbstractMatrix{<:RealU},
    image::AbstractArray{<:RealU, 3},
    plan::SPECTplan,
    thid::Int,
    viewidx::Int,
)
    # rotate image and mumap using multiple processors

#@show thid

    for z in 1:plan.imgsize[3] # 1:nz
        # rotate image in plan.imgr
        imrotate!(
            (@view plan.imgr[thid][:, :, z]),
            (@view image[:, :, z]),
            plan.viewangle[viewidx],
            plan.planrot[thid],
        )
        # rotate mumap and store in plan.mumapr
        imrotate!(
            (@view plan.mumapr[thid][:, :, z]),
            (@view plan.mumap[:, :, z]),
            plan.viewangle[viewidx],
            plan.planrot[thid],
        )
    end

    for y in 1:plan.imgsize[2] # 1:ny
        # account for half of the final slice thickness
        scale3dj!(plan.exp_mumapr[thid], plan.mumapr[thid], y, -0.5)

        for j in 1:y
            plus3dj!(plan.exp_mumapr[thid], plan.mumapr[thid], j)
        end

        broadcast!(*, plan.exp_mumapr[thid], plan.exp_mumapr[thid], - plan.dy)

        broadcast!(exp, plan.exp_mumapr[thid], plan.exp_mumapr[thid])

        # apply depth-dependent attenuation
        mul3dj!(plan.imgr[thid], plan.exp_mumapr[thid], y)

        fft_conv!(
            (@view plan.add_img[thid][:, y, :]),
            (@view plan.imgr[thid][:, y, :]),
            (@view plan.psfs[:, :, y, viewidx]),
            plan.planpsf[thid],
        )

    end

    copy3dj!(view, plan.add_img[thid], 1) # initialize accumulator
    for y in 2:plan.imgsize[2] # accumulate to get total view
        plus3dj!(view, plan.add_img[thid], y)
    end

    # plan.add_img[2] # why does julia allocate (on heap!?) here?

    return view
end


"""
    project!(views, image, plan; index)
Project `image` into multiple `views` with indexes `index` (defaults to `1:nview`).
The 3D `views` array must be pre-allocated, but need not be initialized.
"""
function project!(
    views::AbstractArray{<:RealU,3},
    image::AbstractArray{<:RealU,3},
    plan::SPECTplan;
    index::AbstractVector{<:Int} = 1:plan.nview, # all views
)

    # loop over each view index
    if plan.mode === :fast
        Threads.@threads for (i, viewidx) in collect(enumerate(index))
            thid = Threads.threadid() # todo NO!
            project!((@view views[:,:,i]), image, plan, thid, viewidx)
        end # COV_EXCL_LINE
    else
        for (i, viewidx) in collect(enumerate(index))
            project!((@view views[:,:,i]), image, plan, viewidx)
        end
    end

    return views
end


"""
    views = project(image, plan ; kwargs...)
Convenience method for SPECT forward projector that allocates and returns views.
"""
function project(
    image::AbstractArray{<:RealU,3},
    plan::SPECTplan;
    kwargs...,
)
    views = Array{plan.T}(undef, plan.imgsize[1], plan.imgsize[3], plan.nview)
    project!(views, image, plan; kwargs...)
    return views
end


"""
    views = project(image, mumap, psfs, dy; interpmeth, kwargs...)
Convenience method for SPECT forward projector that does all allocation
including initializing `plan`.

In
* `image` : 3D array `(nx,ny,nz)`
* `mumap` : `(nx,ny,nz)` 3D attenuation map, possibly zeros()
* `psfs` : 4D PSF array
* `dy::RealU` : pixel size
Option
* `interpmeth` : `:one` or `:two`
"""
function project(
    image::AbstractArray{<:RealU, 3},
    mumap::AbstractArray{<:RealU, 3}, # (nx,ny,nz) 3D attenuation map
    psfs::AbstractArray{<:RealU, 4}, # (px,pz,ny,nview)
    dy::RealU;
    interpmeth::Symbol = :two,
    mode::Symbol = :fast,
#   nthread::Int = Threads.nthreads(), # todo: option for plan
    kwargs...,
)
    size(mumap) == size(image) || throw(DimensionMismatch("image/mumap size"))
    size(image,2) == size(psfs,3) || throw(DimensionMismatch("image/psfs size"))
    plan = SPECTplan(mumap, psfs, dy; interpmeth, mode, kwargs...)
    return project(image, plan; kwargs...)
end
