# plan-psf.jl

export PlanPSF, plan_psf

using FFTW: plan_fft!, plan_ifft!


"""
    PlanPSF{T,Tf,Ti}( ; nx::Int, nz::Int, px::Int, pz::Int, T::Type)
Struct for storing work arrays and factors for 2D convolution for one thread.
Each PSF is `px × pz`
- `T` datatype of work arrays (subtype of `AbstractFloat`)
- `nx::Int = 128` (`ny` implicitly the same)
- `nz::Int = nx` image size is `[nx,nx,nz]`
- `px::Int = 1`
- `pz::Int = px` (PSF size)
- `padsize::Tuple` : `(padup, paddown, padleft, padright)`
- `workmat [nx+padup+paddown, nz+padleft+padright]` 2D padded image for FFT convolution
- `workvecx [nx+padup+paddown,]`: 1D work vector
- `workvecz [nz+padleft+padright,]`: 1D work vector
- `img_compl [nx+padup+paddown, nz+padleft+padright]`: 2D [complex] padded image for FFT
- `ker_compl [nx+padup+paddown, nz+padleft+padright]`: 2D [complex] padded image for FFT
- `fft_plan::Tf` plan for doing FFT; see `plan_fft!`
- `ifft_plan::Ti` plan for doing IFFT; see `plan_ifft!`
"""
struct PlanPSF{T, Tf, Ti}
    nx::Int
    nz::Int
    px::Int
    pz::Int
    padsize::NTuple{4, Int}
    workmat::Matrix{T}
    workvecx::Vector{T}
    workvecz::Vector{T}
    img_compl::Matrix{Complex{T}}
    ker_compl::Matrix{Complex{T}}
    fft_plan::Tf # Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}
    ifft_plan::Ti # Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}

    function PlanPSF( ;
        nx::Int = 128,
        nz::Int = nx,
        px::Int = 1,
        pz::Int = px,
        T::Type{<:AbstractFloat} = Float32,
    )

        T <: AbstractFloat || throw("invalid T=$T")
        padup = _padup(nx, px)
        paddown = _paddown(nx, px)
        padleft = _padleft(nz, pz)
        padright = _padright(nz, pz)
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
        Tf = typeof(fft_plan)
        Ti = typeof(ifft_plan)

        new{T, Tf, Ti}(
            nx,
            nz,
            px,
            pz,
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
    plan_psf( ; nx::Int, nz::Int, px::Int, pz::Int, nthread::Int, T::Type)
Make Vector of structs for storing work arrays and factors
for 2D convolution with SPECT depth-dependent PSF model,
threaded across planes parallel to detector.
Option
- `nx::Int = 128`
- `nz::Int = nx`
- `px::Int = 1`
- `pz::Int = px` PSF size is `px × pz`
- `T` : datatype of work arrays, defaults to `Float32`
- `nthread::Int` # of threads, defaults to `Threads.nthreads()`
"""
function plan_psf( ;
    nx::Int = 128,
    nz::Int = nx,
    px::Int = 1,
    pz::Int = px,
    nthread::Int = Threads.nthreads(),
    T::Type{<:AbstractFloat} = Float32,
)
    return [PlanPSF( ; nx, nz, px, pz, T) for id in 1:nthread]
end


"""
    show(io::IO, ::MIME"text/plain", plan::PlanPSF)
"""
function Base.show(io::IO, ::MIME"text/plain", plan::PlanPSF{T}) where {T}
    t = typeof(plan)
    println(io, t)
    for f in (:nx, :nz, :px, :pz, :padsize)
        p = getfield(plan, f)
        t = typeof(p)
        println(io, " ", f, "::", t, " ", p)
    end
    for f in (:workmat, :workvecx, :workvecz, :img_compl, :ker_compl, :fft_plan, :ifft_plan)
        p = getfield(plan, f)
        println(io, " ", f, ":", " ", summary(p))
    end
    println(io, " (", sizeof(plan), " bytes)")
end


"""
    show(io::IO, mime::MIME"text/plain", vp::Vector{<:PlanPSF})
"""
function Base.show(io::IO, mime::MIME"text/plain", vp::Vector{<: PlanPSF})
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
