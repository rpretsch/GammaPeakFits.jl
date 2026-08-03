"""
Module for Bayesian fitting of gamma-ray peaks in a pre-binned energy spectrum.

Provides parametric models for gamma-peak shapes (Gaussian core, Compton edge, ex-Gaussian 
tails) and polynomial backgrounds, along with utilities to construct `BAT.jl` priors and 
posteriors.

# Exports

## Types
- Component parameters: [`GaussianParams`](@ref), [`ComptonParams`](@ref),
  [`ExGaussianParams`](@ref), [`QuadPolyParams`](@ref), [`LinPolyParams`](@ref),
  [`ConstPolyParams`](@ref)
- Containers: [`PeakParams`](@ref), [`BackgroundParams`](@ref), [`ModelParams`](@ref)
- Data: [`SpectrumData`](@ref)

## Model evaluation
- Components: [`gaussian`](@ref), [`compton`](@ref), [`exGaussian`](@ref),
  [`quad_polynomial`](@ref), [`lin_polynomial`](@ref), [`const_polynomial`](@ref)
- Combined models: [`peak_model`](@ref), [`background_model`](@ref), [`full_model`](@ref)

## Fitting
- [`poisson_ll`](@ref), [`build_prior`](@ref), [`build_posterior`](@ref)

## Utils
- [`plot_data`](@ref), [`cut_data`](@ref), [`get_peak_features`](@ref)


# Quick start

```julia
# Fitting constants that have to be provided, but still will be fitted around the here
# specified value
MU = 2048.0 # keV
SIGMA = 10.0 # keV

# Generate data
A = 1000.0              # counts
C_const = 100.0         # counts/keV
lower_limit = 1.0       # keV
upper_limit = 4096.0    # keV
bin_size = 0.5          # keV

gaussian_params = GaussianParams(A = A, mu = MU, sigma = SIGMA)
peak_params = PeakParams(gaussian = gaussian_params)
constPoly_params = ConstPolyParams(C = C_const)
background_params = BackgroundParams(constPoly = constPoly_params)
generation_modelParams = ModelParams(peak = peak_params, background = background_params)
data = SpectrumData(lower_limit, upper_limit, bin_size, generation_modelParams)

# or use existing data instead
# data = SpectrumData(
#            bin_centers = loaded_binCenters,   # keV
#            weights = loaded_weights,          # counts/bin
#            bin_size = loaded_binSize,         # keV
#        )

# cut appropriate fit window
window_size = 50.0 # keV
fit_data = cut_data(data, MU, window_size)

# Specify which components to include for fitting
peak_params = PeakParams(gaussian = true)
background_params = BackgroundParams(constPoly = true)
fit_modelParams = ModelParams(peak = peak_params, background = background_params)

# Get needed peak features
peak_height, peak_area = get_peak_features(fit_data, MU, SIGMA) # (counts/keV, counts)

# Build the prior
prior = build_prior(
            fit_modelParams, 
            MU, 
            SIGMA;
            peak_height = peak_height, 
            peak_area = peak_area
        )

# Build the posterior
posterior = build_posterior(fit_data, prior)

# Sample with BAT.jl
# result = bat_sample(
#              posterior, 
#              TransformedMCMC(proposal=RandomWalk(), nsteps=10^5, nchains=4)
#          )
```
"""
module GammaPeakFits

using BAT
using CairoMakie
using Distributions
using QuadGK: quadgk
using SpecialFunctions: erfc, logerfcx
using ValueShapes: NamedTupleDist

include("types.jl")
include("models.jl")
include("fitting.jl")
include("utils.jl")

# Types — component parameters
export GaussianParams
export ComptonParams
export ExGaussianParams
export QuadPolyParams
export LinPolyParams
export ConstPolyParams

# Types — containers
export PeakParams
export BackgroundParams
export ModelParams

# Types — data
export SpectrumData

# Model evaluation — components
export gaussian
export compton
export exGaussian
export quad_polynomial
export lin_polynomial
export const_polynomial

# Model evaluation — combined
export peak_model
export background_model
export full_model

# Fitting
export poisson_ll
export build_prior
export build_posterior

# Utils
export plot_data
export cut_data
export get_peak_features

end
