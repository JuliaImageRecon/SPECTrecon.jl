# plan-psf.jl

export PlanPSF, plan_psf
import AbstractFFTs
import FFTW

"""
    PlanPSF(nx::Int, nz::Int, nx_psf::Int; T::DataType)
Make struct for storing work arrays and factors for 2D convolution for one thread
Currently PSF must be square, i.e., `nx_psf` = `nz_psf`
- `T` datatype of work arrays
- `nx` must be Int
- `nz` must be Int
- `nx_psf` must be Int
- `padsize{padup, paddown, padleft, padright}` Tuple of padsize
- `workmat [nx+padup+paddown, nz+padleft+padright]` 2D padded image for FFT convolution
- `workvecx [nx+padup+paddown,]`: 1D work vector
- `workvecz [nz+padleft+padright,]`: 1D work vector
- `img_compl [nx+padup+paddown, nz+padleft+padright]`: 2D [complex] padded image for fft
- `ker_compl [nx+padup+paddown, nz+padleft+padright]`: 2D [complex] padded image for fft
- `fft_plan` plan for doing fft, see plan_fft!
- `ifft_plan` plan for doing ifft, see plan_ifft!
"""
struct PlanPSF{T}
    nx::Int
    nz::Int
    nx_psf::Int
    padsize::NTuple{4, Int}
    workmat::Matrix{T}
    workvecx::Vector{T}
    workvecz::Vector{T}
    img_compl::Matrix{Complex{T}}
    ker_compl::Matrix{Complex{T}}
    fft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}
    ifft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}

    function PlanPSF(
        nx::Int,
        nz::Int,
        nx_psf::Int;
        T::DataType = Float32,
    )

        padup = _padup(nx, nx_psf)
        paddown = _paddown(nx, nx_psf)
        padleft = _padleft(nz, nx_psf) # nx_psf = nz_psf!
        padright = _padright(nz, nx_psf) # nx_psf = nz_psf!
        padsize = (padup, paddown, padleft, padright)

        workmat = Matrix{T}(undef, nx+padup+paddown, nz+padleft+padright)
        workvecx = Vector{T}(undef, nx+padup+paddown)
        workvecz = Vector{T}(undef, nz+padleft+padright)

        # complex padimg
        img_compl = Matrix{Complex{T}}(undef, nx+padup+paddown, nz+padleft+padright)
        # complex kernel
        ker_compl = Matrix{Complex{T}}(undef, nx+padup+paddown, nz+padleft+padright)

        fft_plan = plan_fft!(ker_compl)
        ifft_plan = plan_ifft!(ker_compl)

        new{T}(nx,
               nz,
               nx_psf,
               padsize,
               workmat,
               workvecx,
               workvecz,
               img_compl,
               ker_compl,
               fft_plan,
               ifft_plan,
               )
    end
end


"""
    plan_psf(nx::Int, nz::Int, nx_psf::Int; nthread::Int, T::DataType)
Make Vector of structs for storing work arrays and factors for 2D convolution
Input
- `nx::Int`
- `nz::Int`
- `nx_psf::Int`
Option
- `T` : datatype of work arrays, defaults to `Float32`
- `nthread::Int` # of threads, defaults to `Threads.nthreads()`
"""
function plan_psf(
    nx::Int,
    nz::Int,
    nx_psf::Int;
    nthread::Int = Threads.nthreads(),
    T::DataType = Float32,
    )
    return [PlanPSF(nx, nz, nx_psf; T) for id = 1:nthread]
end


"""
    show(io::IO, ::MIME"text/plain", plan::PlanPSF)
"""
function Base.show(io::IO, ::MIME"text/plain", plan::PlanPSF{T}) where {T}
    t = typeof(plan)
    println(io, t)
    for f in (:nx, :nz, :nx_psf, :padsize)
        p = getproperty(plan, f)
        t = typeof(p)
        println(io, " ", f, "::", t, " ", p)
    end
    for f in (:workmat, :workvecx, :workvecz, :img_compl, :ker_compl, :fft_plan, :ifft_plan)
        p = getproperty(plan, f)
        println(io, " ", f, ":", " ", summary(p))
    end
    println(io, " (", sizeof(plan), " bytes)")
end


"""
    show(io::IO, mime::MIME"text/plain", vp::Vector{<:PlanPSF})
"""
function Base.show(io::IO, mime::MIME"text/plain", vp::Vector{PlanPSF{T}}) where {T}
    t = typeof(vp)
    println(io, length(vp), "-element ", t, " with N=", vp[1].nx)
#   show(io, mime, vp[1])
end


"""
    sizeof(::PlanPSF)
Show size in bytes of `PlanPSF` object.
"""
function Base.sizeof(ob::T) where {T <: PlanPSF}
    sum(f -> sizeof(getfield(ob, f)), fieldnames(typeof(ob)))
end
