using SPECTrecon: SPECTrecon
import Aqua
using Test: @testset

@testset "aqua" begin
    Aqua.test_all(
        SPECTrecon;
        deps_compat = (; ignore = [:LinearAlgebra]),
    )
end
