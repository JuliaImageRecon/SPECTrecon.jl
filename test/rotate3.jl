using Main.SPECTrecon:imrotate3!, imrotate3_adj!, imrotate3emmt!, imrotate3emmt_adj!
using Test: @test, @testset, @test_throws, @inferred

@testset "imrotate3" begin
    M = 16
    N = 16
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    x = randn(Float32, M, N)
    y = randn(Float32, M, N)
    output_x = OffsetArrays.no_offset_view(padarray(x, Fill(0, (pad_x, pad_y))))
    output_y = OffsetArrays.no_offset_view(padarray(y, Fill(0, (pad_x, pad_y))))
    tmp_x = similar(output_x)
    tmp_y = similar(output_y)
    θ_list = [π/7, 3π/7, 5π/7, π, 9π/7, 11π/7, 13π/7]
    for θ in θ_list
        @test isapprox(vdot(y, imrotate3!(output_x, tmp_x, x, θ, M, N, pad_x, pad_y)),
                    vdot(x, imrotate3_adj!(output_y, tmp_y, y, θ, M, N, pad_x, pad_y)))
        @test isapprox(vdot(y, imrotate3emmt!(output_x, tmp_x, x, θ, M, N, pad_x, pad_y)),
                    vdot(x, imrotate3emmt_adj!(output_y, tmp_y, y, θ, M, N, pad_x, pad_y)))
    end
end
