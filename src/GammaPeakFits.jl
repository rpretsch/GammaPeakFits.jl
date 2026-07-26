"""
    GammaPeakFits
"""
module GammaPeakFits

using BAT
using Distributions
using QuadGK: quadgk
using SpecialFunctions: erfc

include("types.jl")
include("models.jl")
include("fitting.jl")

end
