using GammaPeakFits
using Test

using BAT: PosteriorMeasure
using Distributions
using SpecialFunctions: erfc, erfcx
using ValueShapes: NamedTupleDist

@testset "GammaPeakFits.jl" begin

    if isempty(ARGS) || "Aqua" in ARGS
        include("aqua.jl")
    end

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
