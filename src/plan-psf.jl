# plan-psf.jl

export PlanPSF, plan_psf
import AbstractFFTs
import FFTW

"""
    PlanPSF{T,Tf,Ti}(nx::Int, nz::Int, nx_psf::Int; T::DataType)
Struct for storing work arrays and factors for 2D convolution for one thread.
Currently PSF must be square, i.e., `nx_psf` = `nz_psf`
- `T` datatype of work arrays (subtype of `AbstractFloat`)
- `nx::Int` (`ny` implicitly the same)
- `nz::Int` image size is `[nx,nx,nz]`
- `nx_psf::Int`
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
    nx_psf::Int
    padsize::NTuple{4, Int}
    workmat::Matrix{T}
    workvecx::Vector{T}
    workvecz::Vector{T}
    img_compl::Matrix{Complex{T}}
    ker_compl::Matrix{Complex{T}}
    fft_plan::Tf # Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}
    ifft_plan::Ti # Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}

    function PlanPSF(
        nx::Int,
        nz::Int,
        nx_psf::Int;
        T::DataType = Float32,
    )

        T <: AbstractFloat || throw("invalid T=$T")
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
        Tf = typeof(fft_plan)
        Ti = typeof(ifft_plan)

        new{T, Tf, Ti}(
            nx,
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
Make Vector of structs for storing work arrays and factors
for 2D convolution with SPECT depth-dependent PSF model,
threaded across planes parallel to detector.
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
