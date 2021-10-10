# backproject.jl

export backproject, backproject!

"""
    backproject!(image, view, plan, viewidx)
Backproject a single view.
"""
function backproject!(
    image::AbstractArray{<:RealU, 3},
    view::AbstractMatrix{<:RealU},
    plan::SPECTplan,
    viewidx::Int,
)

    Threads.@threads for z = 1:plan.imgsize[3] # 1:nz
        thid = Threads.threadid() # thread id

        # rotate mumap
        imrotate!((@view plan.mumapr[:, :, z]),
                  (@view plan.mumap[:, :, z]),
                  plan.viewangle[viewidx],
                  plan.planrot[thid],
                  )

    end

    # adjoint of convolving img with psf and applying attenuation map
    Threads.@threads for y = 1:plan.imgsize[2] # 1:ny
        thid = Threads.threadid() # thread id
        # account for half of the final slice thickness
        scale3dj!(plan.exp_mumapr[thid], plan.mumapr, y, -0.5)
        for j = 1:y
            plus3dj!(plan.exp_mumapr[thid], plan.mumapr, j)
        end

        broadcast!(*, plan.exp_mumapr[thid], plan.exp_mumapr[thid], - plan.dy)

        broadcast!(exp, plan.exp_mumapr[thid], plan.exp_mumapr[thid])

        fft_conv_adj!((@view plan.imgr[:, y, :]),
                       view,
                       (@view plan.psfs[:, :, y, viewidx]),
                       plan.planpsf[thid],
                       )

        mul3dj!(plan.imgr, plan.exp_mumapr[thid], y)
    end

    # adjoint of rotating image
    Threads.@threads for z = 1:plan.imgsize[3] # 1:nz
        thid = Threads.threadid()

        imrotate_adj!((@view image[:, :, z]),
                      (@view plan.imgr[:, :, z]),
                      plan.viewangle[viewidx],
                      plan.planrot[thid],
                      )
    end

    return image
end



"""
    backproject!(image, views, plan ; index)
Backproject multiple views into `image`.
Array `image` is not initialized to zero; caller must do that.
"""
function backproject!(
    image::AbstractArray{<:RealU, 3},
    views::AbstractArray{<:RealU, 3},
    plan::SPECTplan;
    index::AbstractVector{<:Int} = 1:plan.nview, # all views
)

    # loop over each view index
    for i in index
        backproject!(plan.add_img, (@view views[:, :, i]), plan, i)
        broadcast!(+, image, image, plan.add_img)
    end
end



"""
    image = backproject(views, plan ; kwargs...)
SPECT backproject `views`; this allocates the returned 3D array.
"""
function backproject(
    views::AbstractArray{<:RealU, 3},
    plan::SPECTplan;
    kwargs...,
)
    image = zeros(plan.T, plan.imgsize)
    backproject!(image, views, plan; kwargs...)
    return image
end


"""
    image = backproject(views, mumap, psfs, dy; interpmeth, kwargs...)
SPECT backproject `views` using attenuation map `mumap` and PSF array `psfs` for pixel size `dy`.
This method initializes the `plan` as a convenience.
Most users should use `backproject!` instead after initializing those, for better efficiency.
"""
function backproject(
    views::AbstractArray{<:RealU, 3},
    mumap::AbstractArray{<:RealU, 3}, # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:RealU, 4},
    dy::RealU;
    interpmeth::Symbol = :two,
    kwargs...,
)

    plan = SPECTplan(mumap, psfs, dy; interpmeth, kwargs...)
    return backproject(views, plan; kwargs...)
end
