# project.jl


"""
    SPECTplan

Struct for storing key factors for a SPECT system model
- `mumap [nx,ny,nz]` attenuation map, must be 3D, possibly zeros()
- `psfs [n,n,nx,nview]` usually 4D, but could be 3D for a circular orbit

Currently code assumes each of the `nview` projection views is `[nx,nz]`
Currently code assumes `nx = ny`
"""
struct SPECTplan
    mumap::AbstractArray{<:Real} # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:Real} # PSFs could be 3D or 4D
    nview::Int
    work_fft::AbstractMatrix
    # other options for how to do the projection?

    function SPECTplan(mumap, psfs, nview::Int)
        # check nx = ny
        work_fft = zeros(T, size?)
        new(mumap, psfs, nview, work_fft) # todo?
    end
end


# todo: write initializer(s) for SPECTplan


"""
    project!(views, plan, image ; index)
"""
function project!(
    views::AbstractArray{<:AbstractFloat,3},
    plan::SPECTplan,
    image::AbstractArray{<:Real,3} ;
    index::AbstractVector{<:Int} = 1:plan.nview, # all views
)

    # loop over each view index
    for i in index
        project!((@view views[:,:,i]), plan, image)
    end
end



"""
    project!(view, plan, image)
"""
function project!(
    view::AbstractMatrix{<:AbstractFloat},
    plan::SPECTplan,
    image::AbstractArray{<:Real,3},
)
    # todo : read multiple dispatch
 
    # rotate image

    # rotate mumap
    # running sum and exp? of mumap

    # loop over image planes
        # use zero-padded fft (because big) or conv (if small) to convolve with psf
        # sum, account for mumap
end


"""
    views = project(plan, image ; index)
"""
function project(
    plan::SPECTplan,
    image::AbstractArray{<:Real,3} ;
    kwargs...,
)

    nx = size(image, 1)
    nz = size(image, 3)
    views = zeros(eltype(image), nx, nz, plan.nviews)
    project!(views, plan, image ; kwargs...)
end
