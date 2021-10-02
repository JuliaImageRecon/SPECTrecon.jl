# helper.jl

using Main.SPECTrecon: padzero!, padrepl!, pad2sizezero!, pad_it!
using Main.SPECTrecon: fftshift!, ifftshift!, fftshift2!
using Main.SPECTrecon: plus1di!, plus1dj!, plus2di!, plus2dj!
using Main.SPECTrecon: plus3di!, plus3dj!, plus3dk!, scale3dj!, mul3dj!
using Main.SPECTrecon: copy3dj!
using BenchmarkTools
using ImageFiltering: Fill, Pad, BorderArray
using OffsetArrays
using Test: @test, @testset, detect_ambiguities


@testset "padzero!" begin

                T = Float32
                x = randn(T, 7, 5)
                y = randn(T, 3, 3)
                padzero!(x, y, (2, 2, 1, 1))
                z = OffsetArrays.no_offset_view(BorderArray(y, Fill(0, (2, 1), (2, 1))))
                @test isequal(x, z)
                @btime padzero!($x, $y, (2, 2, 1, 1)) # 10.648 ns (0 allocations: 0 bytes)

end


@testset "padrepl!" begin

                T = Float32
                x = randn(T, 10, 9)
                y = randn(T, 5, 4)
                padrepl!(x, y, (1, 4, 3, 2)) # up, down, left, right
                z = OffsetArrays.no_offset_view(BorderArray(y, Pad(:replicate, (1, 3), (4, 2)))) # up, left, down, right
                @test isequal(x, z)
                @btime padrepl!($x, $y, (1, 4, 3, 2)) # 61.129 ns (0 allocations: 0 bytes)

end


@testset "pad2size!" begin

                T = Float32
                ker = reshape(Int16(1):Int16(9), 3, 3)
                padsize = (8, 8)
                z = randn(T, padsize)
                pad2sizezero!(z, ker, padsize)
                tmp = pad_it!(ker, padsize)
                @test isequal(tmp, z)
                @btime pad2sizezero!($z, $ker, $padsize) # 16.899 ns (0 allocations: 0 bytes)
                @btime pad_it!($ker, $padsize) #32.320 ns (0 allocations: 0 bytes)

end


@testset "fftshift" begin

                b = [1 2;3 4]
                x = [b 2*b;3*b 4*b]
                y = similar(x)
                @btime fftshift!($y, $x) # 76.447 ns (0 allocations: 0 bytes)
                @btime ifftshift!($x, $y) # 82.149 ns (0 allocations: 0 bytes)

end


@testset "fftshift2" begin

                T = Float32
                x = randn(T, 120, 128)
                y = similar(x)
                z = similar(x)
                fftshift!(y, x)
                fftshift2!(z, x)
                @test isequal(y, z)
                @btime fftshift!($z, $x) # 3.095 μs (0 allocations: 0 bytes)
                @btime fftshift2!($z, $x) # 2.776 μs (0 allocations: 0 bytes)

end


@testset "plus1di" begin

                T = Float32
                x = randn(T, 4, 64)
                v = randn(T, 64)
                y = x[2, :] .+ v
                plus1di!(x, v, 2)
                @test isequal(x[2, :], y)
                @btime plus1di!($x, $v, 2) # 28.569 ns (0 allocations: 0 bytes)

end


@testset "plus1dj!" begin

                T = Float32
                x = randn(T, 64, 4)
                v = randn(T, 64)
                y = x[:, 2] .+ v
                plus1dj!(x, v, 2)
                @test isequal(x[:, 2], y)
                @btime plus1dj!($x, $v, 2) # 5.257 ns (0 allocations: 0 bytes)

end


@testset "plus2di!" begin

                T = Float32
                x = randn(64)
                v = randn(4, 64)
                y = x .+ v[2, :]
                plus2di!(x, v, 2)
                @test isequal(x, y)
                @btime plus2di!($x, $v, 2) # 28.383 ns (0 allocations: 0 bytes)

end


@testset "plus2dj!" begin

                T = Float32
                x = randn(T, 64)
                v = randn(T, 64, 4)
                y = x .+ v[:, 2]
                plus2dj!(x, v, 2)
                @test isequal(x, y)
                @btime plus2dj!($x, $v, 2) # 6.207 ns (0 allocations: 0 bytes)

end


@testset "plus3di!" begin

                T = Float32
                x = randn(T, 64, 64)
                v = randn(T, 4, 64, 64)
                y = x .+ v[2, :, :]
                plus3di!(x, v, 2)
                @test isequal(x, y)
                @btime plus3di!($x, $v, 2) # 1.711 μs (0 allocations: 0 bytes)

end


@testset "plus3dj!" begin

                T = Float32
                x = randn(T, 64, 64)
                v = randn(T, 64, 4, 64)
                y = x .+ v[:, 2, :]
                plus3dj!(x, v, 2)
                @test isequal(x, y)
                @btime plus3dj!($x, $v, 2) # 304.141 ns (0 allocations: 0 bytes)

end


@testset "plus3dk!" begin

                T = Float32
                x = randn(T, 64, 64)
                v = randn(T, 64, 64, 4)
                y = x .+ v[:, :, 2]
                plus3dk!(x, v, 2)
                @test isequal(x, y)
                @btime plus3dk!($x, $v, 2) # 279.628 ns (0 allocations: 0 bytes)

end


@testset "scale3dj!" begin

                T = Float32
                x = randn(T, 64, 64)
                v = randn(T, 64, 4, 64)
                s = -0.5
                y = s * v[:, 2, :]
                scale3dj!(x, v, 2, s)
                @test isequal(x, y)
                @btime scale3dj!($x, $v, 2, $s) # 408.392 ns (0 allocations: 0 bytes)

end


@testset "mul3dj!" begin

                T = Float32
                x = randn(T, 64, 4, 64)
                v = randn(T, 64, 64)
                y = x[:,2,:] .* v
                mul3dj!(x, v, 2)
                @test isequal(x[:,2,:], y)
                @btime mul3dj!($x, $v, 2) # 15.197 μs (0 allocations: 0 bytes)

end


@testset "copy3dj!" begin

                T = Float32
                x = randn(T, 64, 64)
                v = randn(T, 64, 4, 64)
                y = v[:,2,:]
                copy3dj!(x, v, 2)
                @test isequal(x, y)
                @btime copy3dj!($x, $v, 2) # 226.772 ns (0 allocations: 0 bytes)

end
