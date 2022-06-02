# mlem.jl
# ML-EM algortihm for SPECT reconstruction
export mlem, mlem!, osem, osem!
export Ablock
using LinearMapsAA: LinearMapAO, LinearMapAA
const AbstractNumber = Union{Number, AbstractArray}

"""
    mlem(x0, ynoisy, background, A; niter = 20)
ML-EM algorithm for SPECT reconstruction.
- `x0`: Initial guess
- `ynoisy`: (Noisy) measurements
- `background`: Background effects, e.g., scatters
- `A`: System matrix
- `niter`: Number of iterations
"""
function mlem(x0::AbstractNumber,
              ynoisy::AbstractNumber,
              background::AbstractNumber,
              A::Union{AbstractArray, LinearMapAO};
              niter::Int = 20)

    all(>(0), background) || throw("need background > 0")
    x = copy(x0)
    asum = A' * ones(eltype(ynoisy), size(ynoisy))
    asum[(asum .== 0)] .= Inf
    time0 = time()
    for iter = 1:niter
        @show iter, extrema(x), time() - time0
        ybar = A * x .+ background # forward model
        x .*= (A' * (ynoisy ./ ybar)) ./ asum # multiplicative update
    end
    return x
end


"""
    mlem!(x, ynoisy, background, A; niter = 20)
Inplace version of ML-EM algorithm for SPECT reconstruction.
- `x`: Current iteration estimate
- `ynoisy`: (Noisy) measurements
- `background`: Background effects, e.g., scatters
- `A`: System matrix
- `niter`: Number of iterations
"""
function mlem!(x::AbstractNumber,
               ynoisy::AbstractNumber,
               background::AbstractNumber,
               A::Union{AbstractArray, LinearMapAO};
               niter::Int = 20)
    all(>(0), background) || throw("need background > 0")
    asum = A' * ones(eltype(ynoisy), size(ynoisy)) # this allocates
    asum[(asum .== 0)] .= Inf
    ybar = similar(ynoisy)
    yratio = similar(ynoisy)
    back = similar(x)
    time0 = time()
    for iter = 1:niter
        @show iter, extrema(x), time() - time0
        mul!(ybar, A, x)
        @. yratio = ynoisy / (ybar + background) # coalesce broadcast!
        mul!(back, A', yratio) # back = A' * (ynoisy / ybar)
        @. x *= back / asum # multiplicative update
    end
    return x
end


"""
    Ablock(plan, nblocks)
Generate a vector of linear maps for OSEM.
-`plan`: SPECTrecon plan
-`nblocks`: Number of blocks in OSEM
"""
function Ablock(plan::SPECTplan, nblocks::Int)
    nx, ny, nz = size(plan.mumap)
    nview = plan.nview
    @assert rem(nview, nblocks) == 0 || throw("nview must be divisible by nblocks!")
    Ab = Vector{LinearMapAO}(undef, nblocks)
    for nb = 1:nblocks
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
    osem(x0, ynoisy, background, Ab; niter = 20)
OS-EM algorithm for SPECT reconstruction.
- `x0`: Initial guess
- `ynoisy`: (Noisy) measurements
- `background`: Background effects, e.g., scatters
- `Ab`: Vector of system matrix
- `niter`: Number of iterations
"""
function osem(x0::AbstractNumber,
              ynoisy::AbstractNumber,
              background::AbstractNumber,
              Ab::Union{Vector{AbstractArray}, Vector{LinearMapAO}};
              niter::Int = 16)

    all(>(0), background) || throw("need background > 0")
    x = copy(x0)
	nx, nz, nview = size(ynoisy)
	nblocks = length(Ab)
	asum = Vector{Array{eltype(ynoisy), 3}}(undef, nblocks)
	for nb = 1:nblocks
	    asum[nb] = Ab[nb]' * ones(eltype(ynoisy), nx, nz, nview÷nblocks)
	    (asum[nb])[(asum[nb] .== 0)] .= Inf # avoid divide by zero
	end
	time0 = time()
    for iter = 1:niter
	    for nb = 1:nblocks
	        @show iter, nb, extrema(x), time() - time0
	        ybar = Ab[nb] * x .+ (@view background[:,:,nb:nblocks:nview]) # forward model
	        x .*= (Ab[nb]' * ((@view ynoisy[:,:,nb:nblocks:nview]) ./ ybar)) ./ asum[nb] # multiplicative update
	    end
    end
    return x
end


"""
    osem!(x, ynoisy, background, Ab; niter = 20)
OS-EM algorithm for SPECT reconstruction.
- `x`: Current iteration estimate
- `ynoisy`: (Noisy) measurements
- `background`: Background effects, e.g., scatters
- `Ab`: Vector of system matrix
- `niter`: Number of iterations
"""
function osem!(x::AbstractNumber,
               ynoisy::AbstractNumber,
               background::AbstractNumber,
               Ab::Union{Vector{AbstractArray}, Vector{LinearMapAO}};
               niter::Int = 16)
    all(>(0), background) || throw("need background > 0")
	nx, nz, nview = size(ynoisy)
	nblocks = length(Ab)
	asum = Vector{Array{eltype(ynoisy), 3}}(undef, nblocks)
	for nb = 1:nblocks
	    asum[nb] = Ab[nb]' * ones(eltype(ynoisy), nx, nz, nview÷nblocks)
        (asum[nb])[(asum[nb] .== 0)] .= Inf # avoid divide by zero
	end
    ybar = Array{eltype(ynoisy)}(undef, nx, nz, nview÷nblocks)
    yratio = similar(ybar)
    back = similar(x)
	time0 = time()
    for iter = 1:niter
        for nb = 1:nblocks
	        @show iter, nb, extrema(x), time() - time0
	        mul!(ybar, Ab[nb], x)
	        @. yratio = (@view ynoisy[:,:,nb:nblocks:nview]) /
			             (ybar + (@view background[:,:,nb:nblocks:nview]))
	        mul!(back, Ab[nb]', yratio) # back = A' * (ynoisy / ybar)
	        @. x *= back / asum[nb] # multiplicative update
	    end
    end
    return x
end
