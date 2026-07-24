"""
    GammaPeakFits
"""
module GammaPeakFits

using Distributions
using QuadGK: quadgk
using SpecialFunctions: erfc

include("types.jl")
include("models.jl")
include("likelihoods.jl")

end
