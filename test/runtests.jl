# runtests.jl — GammaPeakFits test suite.
#
# Recommended (activates test/Project.toml, matching CI):
#     julia --project=. -e 'using Pkg; Pkg.test()'                        # full suite
#     julia --project=. -e 'using Pkg; Pkg.test(; test_args=["models"])'  # selected suites

using GammaPeakFits
using Test

using BAT: PosteriorMeasure
using Distributions
using SpecialFunctions: erfc, logerfcx
using ValueShapes: NamedTupleDist

@testset "GammaPeakFits.jl" begin

    if isempty(ARGS) || "Aqua" in ARGS
        include("aqua.jl")
    end

    if isempty(ARGS) || "params" in ARGS
        include("test_params.jl")
    end

    if isempty(ARGS) || "data" in ARGS
        include("test_data.jl")
    end

    if isempty(ARGS) || "configs" in ARGS
        include("test_configs.jl")
    end

    if isempty(ARGS) || "models" in ARGS
        include("test_models.jl")
    end

    if isempty(ARGS) || "integrals" in ARGS
        include("test_integrals.jl")
    end

    if isempty(ARGS) || "fitting" in ARGS
        include("test_fitting.jl")
    end

    if isempty(ARGS) || "stability" in ARGS
        include("test_stability.jl")
    end

    if isempty(ARGS) || "utils" in ARGS
        include("test_utils.jl")
    end

end
