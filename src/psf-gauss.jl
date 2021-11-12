# psf-gauss.jl

export psf_gauss

"""
    psf_gauss( ; ny, px, pz, fwhm_start, fwhm_end, fwhm, fwhm_x, fwhm_z, T)

Create depth-dependent Gaussian PSFs
having specified full-width half-maximum (FHWM) values.

# Options
- 'ny::Int = 128'
- 'px::Int = 11' (should be odd)
- 'pz::Int = px' (should be odd)
- 'fwhm_start::Real = 1'
- 'fwhm_end::Real = 4'
- 'fwhm::AbstractVector{<:Real} = LinRange(fwhm_start, fwhm_end, ny)'
- 'fwhm_x::AbstractVector{<:Real} = fwhm,
- 'fwhm_z::AbstractVector{<:Real} = fwhm_x'
- 'T::DataType == Float32'

Returned `psf` is `[px, pz, ny]` where each PSF sums to 1.
"""
function psf_gauss( ;
    ny::Int = 128,
    px::Int = 11,
    pz::Int = px,
    fwhm_start::Real = 1,
    fwhm_end::Real = 4,
    fwhm::AbstractVector{<:Real} = LinRange(fwhm_start, fwhm_end, ny),
    fwhm_x::AbstractVector{<:Real} = fwhm,
    fwhm_z::AbstractVector{<:Real} = fwhm_x,
    T::DataType = Float32,
)
    isodd(px) || @warn("even px = $px ?")
    isodd(pz) || @warn("even pz = $pz ?")
    psf = zeros(T, px, pz, ny)

    rx = (-(px-1)÷2):((px-1)÷2)
    rz = (-(pz-1)÷2):((pz-1)÷2)
    for iy in 1:ny # depth-dependent blur
        psf[:,:,iy] = psf_gauss(rx, fwhm_x[iy]) * psf_gauss(rz, fwhm_z[iy])'
    end
    return psf
end

function psf_gauss(r::AbstractVector, fwhm::Real)
   if fwhm == 0
        any(==(0), r) || throw("must have some r=0 if fwhm=0")
        return r .== 0 # Kronecker impulse
   end
   σ = fwhm / sqrt(log(256)) # FWHM to Gaussian σ
   psf = @. exp(-0.5 * abs2(r / σ))
   return psf / sum(psf)
end
