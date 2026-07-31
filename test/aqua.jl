using Aqua

@testset "Aqua.jl quality checks" begin
    Aqua.test_all(GammaPeakFits; ambiguities = true)
end
