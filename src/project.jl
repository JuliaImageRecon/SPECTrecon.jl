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

    Threads.@threads for z = 1:plan.imgsize[3] # 1:nz
        thid = Threads.threadid() # thread id
        # rotate image in plan.imgr
        imrotate!((@view plan.imgr[:, :, z]),
                  (@view image[:, :, z]),
                  plan.viewangle[viewidx],
                  plan.planrot[thid],
                  )
        # rotate mumap and store in plan.mumapr
        imrotate!((@view plan.mumapr[:, :, z]),
                  (@view plan.mumap[:, :, z]),
                  plan.viewangle[viewidx],
                  plan.planrot[thid],
                  )
    end

    Threads.@threads for y = 1:plan.imgsize[2] # 1:ny
        thid = Threads.threadid() # thread id
        # account for half of the final slice thickness
        scale3dj!(plan.exp_mumapr[thid], plan.mumapr, y, -0.5)

        for j = 1:y
            plus3dj!(plan.exp_mumapr[thid], plan.mumapr, j)
        end

        broadcast!(*, plan.exp_mumapr[thid], plan.exp_mumapr[thid], - plan.dy)

        broadcast!(exp, plan.exp_mumapr[thid], plan.exp_mumapr[thid])

        # apply depth-dependent attenuation
        mul3dj!(plan.imgr, plan.exp_mumapr[thid], y)

        fft_conv!((@view plan.add_img[:, y, :]),
                  (@view plan.imgr[:, y, :]),
                  (@view plan.psfs[:, :, y, viewidx]),
                  plan.planpsf[thid],
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
    for i in index
        project!((@view views[:,:,i]), image, plan, i)
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
* `image` : 3D array `[nx,ny,nz]`
* `mumap` : `[nx,ny,nz]` 3D attenuation map, possibly zeros()
* `psfs` : 4D PSF array
* `dy::RealU` : pixel size
Option
* `interpmeth` : :one or :two
"""
function project(
    image::AbstractArray{<:RealU, 3},
    mumap::AbstractArray{<:RealU, 3}, # [nx,ny,nz] 3D attenuation map
    psfs::AbstractArray{<:RealU, 4},
    dy::RealU;
    interpmeth::Symbol = :two,
#   nthread::Int = Threads.nthreads(), # todo: option for plan
    kwargs...,
)
    plan = SPECTplan(mumap, psfs, dy; interpmeth, kwargs...)
    return project(image, plan; kwargs...)
end
