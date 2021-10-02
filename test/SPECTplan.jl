# SPECTplan.jl

using Main.SPECTrecon: SPECTplan, Workarray
using Test: @test, @testset, detect_ambiguities

@testset "spectplan" begin

                T = Float32
                nx = 128
                ny = 128
                nz = 81
                nx_psf = 37
                nz_psf = 37
                dy = T(4.80)
                nview = 120
                mumap = randn(T, nx, ny, nz)
                psfs = ones(T, nx_psf, nz_psf, ny, nview)
                plan = SPECTplan(mumap, psfs, nview, dy)
                workarray = Vector{Workarray}(undef, plan.ncore)
                for i = 1:plan.ncore
                                workarray[i] = Workarray(plan.T, plan.imgsize, plan.pad_fft, plan.pad_rot) # allocate
                end

                plan = SPECTplan(mumap, psfs, nview, dy)
                @btime plan = SPECTplan($mumap, $psfs, $nview, $dy)
                # 57.470 ms (199862 allocations: 102.89 MiB)

                workarray = Vector{Workarray}(undef, plan.ncore)
                @btime workarray = Vector{Workarray}(undef, $plan.ncore)
                # 85.247 ns (1 allocation: 1008 bytes)
                @btime for i = 1:$plan.ncore
                                $workarray[i] = Workarray($plan.T, $plan.imgsize, $plan.pad_fft, $plan.pad_rot) # allocate
                end
                # 1.902 ms (3920 allocations: 7.58 MiB)

end
