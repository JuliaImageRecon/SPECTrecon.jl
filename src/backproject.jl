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

    # rotate image and mumap using multiple processors (adjoint)
    nz = plan.imgsize[3] # prepare to loop over slices
    spawner(plan.nthread, nz) do buffer_id, iz
        # rotate mumap
        imrotate!(
            (@view plan.mumapr[:, :, iz]),
            (@view plan.mumap[:, :, iz]),
            plan.viewangle[viewidx],
            plan.planrot[buffer_id],
        )
    end

    # adjoint of convolving img with psf and applying attenuation map
    ny = plan.imgsize[2] # prepare to loop over y planes
    spawner(plan.nthread, ny) do buffer_id, iy
        # account for half of the final slice thickness
        scale3dj!(plan.exp_mumapr[buffer_id], plan.mumapr, iy, -0.5)

        for j in 1:iy
            plus3dj!(plan.exp_mumapr[buffer_id], plan.mumapr, j)
        end

        broadcast!(*, plan.exp_mumapr[buffer_id], plan.exp_mumapr[buffer_id], - plan.dy)
        broadcast!(exp, plan.exp_mumapr[buffer_id], plan.exp_mumapr[buffer_id])

        fft_conv_adj!(
            (@view plan.imgr[:, iy, :]),
            view,
            (@view plan.psfs[:, :, iy, viewidx]),
            plan.planpsf[buffer_id],
        )

        mul3dj!(plan.imgr, plan.exp_mumapr[buffer_id], iy)
    end

    # adjoint of rotate image
    spawner(plan.nthread, nz) do buffer_id, iz
        imrotate_adj!(
            (@view image[:, :, iz]),
            (@view plan.imgr[:, :, iz]),
            plan.viewangle[viewidx],
            plan.planrot[buffer_id],
        )
    end

    return image
end


"""
    backproject!(image, view, plan, buffer_id, viewidx)
Backproject a single view.
"""
function backproject!(
    image::AbstractArray{<:RealU, 3},
    view::AbstractMatrix{<:RealU},
    plan::SPECTplan,
    buffer_id::Int,
    viewidx::Int,
)

    # rotate mumap
    for z in 1:plan.imgsize[3] # 1:nz
        imrotate!(
            (@view plan.mumapr[buffer_id][:, :, z]),
            (@view plan.mumap[:, :, z]),
            plan.viewangle[viewidx],
            plan.planrot[buffer_id],
        )

    end

    # adjoint of convolving img with psf and applying attenuation map
    for y in 1:plan.imgsize[2] # 1:ny
        # account for half of the final slice thickness
        scale3dj!(plan.exp_mumapr[buffer_id], plan.mumapr[buffer_id], y, -0.5)

        for j in 1:y
            plus3dj!(plan.exp_mumapr[buffer_id], plan.mumapr[buffer_id], j)
        end

        broadcast!(*, plan.exp_mumapr[buffer_id], plan.exp_mumapr[buffer_id], - plan.dy)

        broadcast!(exp, plan.exp_mumapr[buffer_id], plan.exp_mumapr[buffer_id])

        fft_conv_adj!(
            (@view plan.imgr[buffer_id][:, y, :]),
            view,
            (@view plan.psfs[:, :, y, viewidx]),
            plan.planpsf[buffer_id],
        )

        mul3dj!(plan.imgr[buffer_id], plan.exp_mumapr[buffer_id], y)
    end

    # adjoint of rotating image
    for z in 1:plan.imgsize[3] # 1:nz
        imrotate_adj!(
            (@view plan.imgr[buffer_id][:, :, z]),
            (@view plan.imgr[buffer_id][:, :, z]),
            plan.viewangle[viewidx],
            plan.planrot[buffer_id],
        )
    end

    broadcast!(+, image, image, plan.imgr[buffer_id])

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
    image .= zero(plan.T) # must be initialized as zero
    if plan.mode === :fast
        for i in 1:plan.nthread
             plan.add_img[i] .= zero(plan.T)
        end

        spawner(plan.nthread, length(index)) do buffer_id, ii
            viewidx = index[ii]
            backproject!(plan.add_img[buffer_id], (@view views[:,:,ii]), plan, buffer_id, viewidx)
        end

        for i in 1:plan.nthread
            broadcast!(+, image, image, plan.add_img[i])
        end
    else
        for (i, viewidx) in collect(enumerate(index))
            backproject!(plan.add_img, (@view views[:, :, i]), plan, viewidx)
            broadcast!(+, image, image, plan.add_img)
        end
    end

    return image
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
    views::AbstractArray{<:RealU, 3}, # [nx,nz,nview]
    mumap::AbstractArray{<:RealU, 3}, # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:RealU, 4},
    dy::RealU;
    interpmeth::Symbol = :two,
    mode::Symbol = :fast,
#   nthread::Int = Threads.nthreads(), # todo: option for plan
    kwargs...,
)

    size(mumap,1) == size(mumap,1) == size(views,1) ||
        throw(DimensionMismatch("nx"))
    size(mumap,3) == size(views,2) || throw(DimensionMismatch("nz"))
    plan = SPECTplan(mumap, psfs, dy; interpmeth, mode, kwargs...)
    return backproject(views, plan; kwargs...)
end
