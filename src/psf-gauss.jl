# psf-gauss.jl

export psf_gauss

"""
    psf_gauss( ; nx, nx_psf, fwhm_start, fwhm_end, fwhm, T)

Create depth-dependent Gaussian PSFs
having specified full-width half-maximum (FHWM) values.

# Options
- 'nx::Int = 128'
- 'nx_psf::Int = 11' (should be odd)
- 'fwhm_start::Real = 1'
- 'fwhm_end::Real = 4'
- 'fwhm::AbstractVector{<:Real} = LinRange(fwhm_start, fwhm_end, nx)'
- 'T::DataType == Float32'

Returned `psf` is `[nx_psf, nx_psf, nx]` where each PSF sums to 1.
"""
function psf_gauss( ;
    nx::Int = 128,
    nx_psf::Int = 11,
    fwhm_start::Real = 1,
    fwhm_end::Real = 4,
    fwhm::AbstractVector{<:Real} = LinRange(fwhm_start, fwhm_end, nx),
    T::DataType = Float32,
)
    isodd(nx_psf) || @warn("even nx_psf = $nx_psf ?")
    psf = zeros(T, nx_psf, nx_psf, nx)

    for iy in 1:nx # depth-dependent blur
        r = (-(nx_psf-1)÷2):((nx_psf-1)÷2)
        σ = fwhm[iy] / sqrt(log(256))
        r2 = abs2.(r / σ)
        tmp = @. exp(-π * (r2 + r2'))
        psf[:,:,iy] = tmp / sum(tmp)
    end
    return psf
end
