"""
    SPECTrecon
System matrix (forward and back-projector) for SPECT image reconstruction.
"""
module SPECTrecon

    const RealU = Number # Union{Real, Unitful.Length}

    include("helper.jl")
    include("plan-rotate.jl")
    include("rotatez.jl")
    include("plan-psf.jl")
    include("fft_convolve.jl")
    include("spectplan.jl")
    include("project.jl")
    include("backproject.jl")

end # module
