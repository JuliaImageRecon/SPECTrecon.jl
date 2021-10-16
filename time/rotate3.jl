# rotate3.jl
# time 3D image rotation using @thread and foreach

using Revise
using SPECTrecon
using BenchmarkTools
using Plots; default(markerstrokecolor=:auto)

nx, nz = 128, 100
T = Float32
image3 = randn(T, nx,nx,nz)
out1 = similar(image3)
out2 = similar(image3)

#plans = plan_rotate(nx; T, nthread = 99) # warns
plans = plan_rotate(nx; T) # default nthread = Threads.nthreads())

θ = π/6
imrotate!(out1, image3, θ, plans, 17) # foreach
imrotate!(out2, image3, θ, plans, :thread)
@assert out1 == out2

#@btime imrotate!($out1, $image3, $θ, $plans, 20) # 5.9 ms (888 allocations: 63.53 KiB)
#@btime imrotate!($out1, $image3, $θ, $plans) # 6.1 ms (802 allocations: 56.77 KiB)
#@btime imrotate!($out2, $image3, $θ, $plans, :thread) # 5.8 ms (41 allocations: 3.17 KiB)
nothing

if !@isdefined(elapse0)
    elapse0 = @belapsed imrotate!($out1, $image3, $θ, $plans, :thread)
end

if !@isdefined(elapse)
	ntask = 1:20
	elapse = zeros(length(ntask))
	for (i,nt) in enumerate(ntask)
		@show i, nt
    	elapse[i] = @belapsed imrotate!($out1, $image3, $θ, $plans, $nt)
	end
end

@show minimum(elapse), elapse0

ncore = Threads.nthreads()
i = findall(==(ncore), ntask)[1]
scatter(ntask, elapse, xlabel="ntask", ylabel="time", label="foreach",
ylim=(0.005,0.03),
xlim=(1,20),
ytick=[0.006,0.010,0.015,0.029],
xtick=2 .^ (0:5),
)
scatter!([ntask[i]], [elapse[i]], color=:red, label="@threads", title="rotate $nx × $nx × $nz")
#savefig("rotate.pdf")
