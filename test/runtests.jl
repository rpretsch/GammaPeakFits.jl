using GammaPeakFits
using Test

using BAT
using Distributions
using SpecialFunctions: erfc
using ValueShapes: NamedTupleDist

@testset "GammaPeakFits.jl" begin
    if isempty(ARGS) || "types" in ARGS
        include("test_types.jl")
    end
    if isempty(ARGS) || "models" in ARGS
        include("test_models.jl")
    end
    if isempty(ARGS) || "fitting" in ARGS
        include("test_fitting.jl")
    end
end
