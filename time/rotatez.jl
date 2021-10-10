# rotatez.jl

using Main.SPECTrecon: plan_rotate
using Main.SPECTrecon: imrotate!, imrotate_adj!
using BenchmarkTools: @btime


function imrotate1_time()
    T = Float32
    N = 100
    img = zeros(T, N, N)
    img[30:50, 20:60] .= 1
    output = similar(img)
    θ = 3*π/16
    plan = plan_rotate(N; nthread = 1, T = T, method = :one)[1]
    println("imrotate1-forw")
    @btime imrotate!($output, $img, $θ, $plan)
    # 533.316 μs (0 allocations: 0 bytes)
    println("imrotate1-back")
    @btime imrotate_adj!($output, $img, $θ, $plan)
    # 517.367 μs (0 allocations: 0 bytes)
    nothing
end


function imrotate2_time()
    T = Float32
    N = 100
    img = zeros(T, N, N)
    img[30:50, 20:60] .= 1
    output = similar(img)
    θ = 3*π/16
    plan = plan_rotate(N; nthread = 1, T = T, method = :two)[1]
    println("imrotate2-forw")
    @btime imrotate!($output, $img, $θ, $plan)
    # 162.171 μs (0 allocations: 0 bytes)
    println("imrotate2-back")
    @btime imrotate_adj!($output, $img, $θ, $plan)
    # 168.518 μs (0 allocations: 0 bytes)
    nothing
end


# run all functions, time may vary on different machines, but should be all zero allocation.
imrotate1_time()
imrotate2_time()
