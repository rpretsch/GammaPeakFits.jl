"""
    GammaPeakFits
"""
module GammaPeakFits

using BAT
using DensityInterface: logfuncdensity
using Distributions
using QuadGK: quadgk
using SpecialFunctions: erfc
using ValueShapes: NamedTupleDist

include("types.jl")
include("models.jl")
include("fitting.jl")

end
