# ml-os-em.jl
# ML-EM algorithm for emission tomography image reconstruction

export mlem, mlem!, osem, osem!
export Ablock
using LinearMapsAA: LinearMapAO, LinearMapAA


"""
    mlem!(out, x0, ynoisy, background, A; niter = 20)
Inplace version of ML-EM algorithm for emission tomography image reconstruction.
- `out`: Output
- `x0`: Initial guess
- `ynoisy`: (Noisy) measurements
- `background`: Background effects, e.g., scatters
- `A`: System matrix
- `niter`: Number of iterations
"""
function mlem!(
    out::AbstractArray,
    x0::AbstractArray,
    ynoisy::AbstractArray,
    background::AbstractArray,
    A::Union{AbstractArray, LinearMapAO};
    niter::Int = 20,
    chat::Bool = false,
 )
    all(>(0), background) || throw("need background > 0")
    size(out) == size(x0) || throw(DimensionMismatch("size out and x0 not match"))
    size(ynoisy) == size(background) || throw(DimensionMismatch("size ynoisy and background"))

    asum = A' * ones(eltype(ynoisy), size(ynoisy)) # this allocates
    asum[(asum .== 0)] .= Inf
    ybar = similar(ynoisy)
    yratio = similar(ynoisy)
    back = similar(x0)
    copyto!(out, x0)
    time0 = time()
    for iter in 1:niter
        chat && (@show iter, extrema(out), time() - time0)
        mul!(ybar, A, out)
        @. yratio = ynoisy / (ybar + background) # coalesce broadcast!
        mul!(back, A', yratio) # back = A' * (ynoisy / ybar)
        @. out *= back / asum # multiplicative update
    end
    return out
end


"""
    mlem(x0, ynoisy, background, A; niter = 20)
ML-EM algorithm for emission tomography image reconstruction.
- `x0`: Initial guess
- `ynoisy`: (Noisy) measurements
- `background`: Background effects, e.g., scatters
- `A`: System matrix
- `niter`: Number of iterations
"""
mlem(x0::AbstractArray, args...; kwargs...) = mlem!(similar(x0), x0, args...; kwargs...)


"""
    Ablock(plan, nblocks)
Generate a vector of linear maps for OSEM.
-`plan`: SPECTrecon plan
-`nblocks`: Number of blocks in OSEM
"""
function Ablock(plan::SPECTplan, nblocks::Int)
    nx, ny, nz = size(plan.mumap)
    nview = plan.nview
    rem(nview, nblocks) == 0 || throw("nview must be divisible by nblocks!")
    Ab = Vector{LinearMapAO}(undef, nblocks)
    for nb in 1:nblocks
        viewidx = nb:nblocks:nview
        forw!(y,x) = project!(y, x, plan; index = viewidx)
        back!(x,y) = backproject!(x, y, plan; index = viewidx)
        idim = (nx,ny,nz)
        odim = (nx,nz,length(viewidx))
        Ab[nb] = LinearMapAA(forw!, back!, (prod(odim),prod(idim)); plan.T, odim, idim)
    end
    return Ab
end


"""
    osem!(out, x0, ynoisy, background, Ab; niter = 20)
OS-EM algorithm for SPECT reconstruction.
- `out`: Output
- `x0`: Initial estimate
- `ynoisy`: (Noisy) measurements
- `background`: Background effects, e.g., scatters
- `Ab`: Vector of system matrix
- `niter`: Number of iterations
"""
function osem!(
    out::AbstractArray,
    x0::AbstractArray,
    ynoisy::AbstractArray,
    background::AbstractArray,
    Ab::Union{Vector{AbstractArray}, Vector{LinearMapAO}};
    niter::Int = 16,
    chat::Bool = false,
)
    all(>(0), background) || throw("need background > 0")
    size(out) != size(x0) && throw(DimensionMismatch("size out and x0 not match"))
    nx, nz, nview = size(ynoisy)
    nblocks = length(Ab)
    asum = Vector{Array{eltype(ynoisy), 3}}(undef, nblocks)
    for nb in 1:nblocks
        asum[nb] = Ab[nb]' * ones(eltype(ynoisy), nx, nz, nview÷nblocks)
        (asum[nb])[asum[nb] .== 0] .= Inf # avoid divide by zero
    end
    ybar = Array{eltype(ynoisy)}(undef, nx, nz, nview÷nblocks)
    yratio = similar(ybar)
    back = similar(x0)
    copyto!(out, x0)
    time0 = time()
    for iter in 1:niter
        for nb in 1:nblocks
	        chat && (@show iter, nb, extrema(out), time() - time0)
	        mul!(ybar, Ab[nb], out)
	        @. yratio = (@view ynoisy[:,:,nb:nblocks:nview]) /
			             (ybar + (@view background[:,:,nb:nblocks:nview]))
	        mul!(back, Ab[nb]', yratio) # back = A' * (ynoisy / ybar)
	        @. out *= back / asum[nb] # multiplicative update
	    end
    end
    return out
end


"""
    osem(x0, ynoisy, background, Ab; niter = 20)
OS-EM algorithm for SPECT reconstruction.
- `x0`: Initial guess
- `ynoisy`: (Noisy) measurements
- `background`: Background effects, e.g., scatters
- `Ab`: Vector of system matrix
- `niter`: Number of iterations
"""
osem(x0::AbstractArray, args...; kwargs...) = osem!(similar(x0), x0, args...; kwargs...)
