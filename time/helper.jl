# helper.jl

using BenchmarkTools: @btime
using SPECTrecon: padzero!, padrepl!, pad2sizezero!, pad_it!
using SPECTrecon: fftshift!, ifftshift!, fftshift2!
using SPECTrecon: plus1di!, plus1dj!, plus2di!, plus2dj!
using SPECTrecon: plus3di!, plus3dj!, plus3dk!, scale3dj!, mul3dj!
using SPECTrecon: copy3dj!


function padzero_time()
    T = Float32
    x = randn(T, 7, 5)
    y = randn(T, 3, 3)
    println("padzero")
    @btime padzero!($x, $y, (2, 2, 1, 1)) # 10.648 ns (0 allocations: 0 bytes)
    nothing
end


function padrepl_time()
    T = Float32
    x = randn(T, 10, 9)
    y = randn(T, 5, 4)
    println("padrepl")
    @btime padrepl!($x, $y, (1, 4, 3, 2)) # 61.129 ns (0 allocations: 0 bytes)
    nothing
end


function pad2size_time()
    T = Float32
    ker = reshape(Int16(1):Int16(9), 3, 3)
    padsize = (8, 8)
    z = randn(T, padsize)
    println("pad2sizezero")
    @btime pad2sizezero!($z, $ker, $padsize) # 16.899 ns (0 allocations: 0 bytes)
    println("pad_it")
    @btime pad_it!($ker, $padsize) #32.320 ns (0 allocations: 0 bytes)
    nothing
end

function fft_time()
    b = [1 2;3 4]
    x = [b 2*b;3*b 4*b]
    y = similar(x)
    println("fftshift")
    @btime fftshift!($y, $x) # 76.447 ns (0 allocations: 0 bytes)
    println("ifftshift")
    @btime ifftshift!($x, $y) # 82.149 ns (0 allocations: 0 bytes)
    nothing
end


function fft2_time()
    T = Float32
    x = randn(T, 120, 128)
    z = similar(x)
    println("fftshift")
    @btime fftshift!($z, $x) # 3.095 μs (0 allocations: 0 bytes)
    println("fftshift2")
    @btime fftshift2!($z, $x) # 2.776 μs (0 allocations: 0 bytes)
    nothing
end


function plus1di_time()
    T = Float32
    x = randn(T, 4, 64)
    v = randn(T, 64)
    println("plus1di")
    @btime plus1di!($x, $v, 2) # 28.569 ns (0 allocations: 0 bytes)
    nothing
end


function plus1dj_time()
    T = Float32
    x = randn(T, 64, 4)
    v = randn(T, 64)
    println("plus1dj")
    @btime plus1dj!($x, $v, 2) # 5.257 ns (0 allocations: 0 bytes)
    nothing
end


function plus2di_time()
    T = Float32
    x = randn(64)
    v = randn(4, 64)
    println("plus2di")
    @btime plus2di!($x, $v, 2) # 28.383 ns (0 allocations: 0 bytes)
    nothing
end


function plus2dj_time()
    T = Float32
    x = randn(T, 64)
    v = randn(T, 64, 4)
    println("plus2dj")
    @btime plus2dj!($x, $v, 2) # 6.207 ns (0 allocations: 0 bytes)
    nothing
end


function plus3di_time()
    T = Float32
    x = randn(T, 64, 64)
    v = randn(T, 4, 64, 64)
    println("plus3di")
    @btime plus3di!($x, $v, 2) # 1.711 μs (0 allocations: 0 bytes)
    nothing
end


function plus3dj_time()
    T = Float32
    x = randn(T, 64, 64)
    v = randn(T, 64, 4, 64)
    println("plus3dj")
    @btime plus3dj!($x, $v, 2) # 304.141 ns (0 allocations: 0 bytes)
    nothing
end


function plus3dk_time()
    T = Float32
    x = randn(T, 64, 64)
    v = randn(T, 64, 64, 4)
    println("plus3dk")
    @btime plus3dk!($x, $v, 2) # 279.628 ns (0 allocations: 0 bytes)
    nothing
end


function scale3dj_time()
    T = Float32
    x = randn(T, 64, 64)
    v = randn(T, 64, 4, 64)
    s = -0.5
    println("scale3dj")
    @btime scale3dj!($x, $v, 2, $s) # 408.392 ns (0 allocations: 0 bytes)
    nothing
end


function mul3dj_time()
    T = Float32
    x = randn(T, 64, 4, 64)
    v = randn(T, 64, 64)
    println("mul3dj")
    @btime mul3dj!($x, $v, 2) # 15.197 μs (0 allocations: 0 bytes)
    nothing
end


function copy3dj_time()
    T = Float32
    x = randn(T, 64, 64)
    v = randn(T, 64, 4, 64)
    println("copy3dj")
    @btime copy3dj!($x, $v, 2) # 226.772 ns (0 allocations: 0 bytes)
    nothing
end


# run all functions, time may vary on different machines, but should be all zero allocation.
padzero_time()
padrepl_time()
pad2size_time()
fft_time()
fft2_time()
plus1di_time()
plus1dj_time()
plus2di_time()
plus2dj_time()
plus3di_time()
plus3dj_time()
plus3dk_time()
scale3dj_time()
mul3dj_time()
copy3dj_time()
