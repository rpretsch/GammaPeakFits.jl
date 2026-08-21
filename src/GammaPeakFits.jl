"""
Module for Bayesian fitting of gamma-ray peaks in a pre-binned energy spectrum.

Provides parametric models for gamma-peak shapes (Gaussian core, Compton edge, ex-Gaussian 
tails) and polynomial backgrounds, along with utilities to construct `BAT.jl` priors and 
posteriors.

# Exports

## Types
- Presence markers: [`AbstractComponent`](@ref), [`Enabled`](@ref), [`Disabled`](@ref)
- Integration methods: [`AbstractIntegrationMethod`](@ref), [`Analytical`](@ref),
  [`Numerical`](@ref), [`Midpoint`](@ref)
- Component parameters: [`GaussianParams`](@ref), [`ComptonParams`](@ref),
  [`ExGaussianParams`](@ref), [`QuadPolyParams`](@ref), [`LinPolyParams`](@ref),
  [`ConstPolyParams`](@ref)
- Containers: [`PeakParams`](@ref), [`BackgroundParams`](@ref), [`ModelParams`](@ref)
- Data: [`SpectrumData`](@ref)
- Configuration: [`FitConfigs`](@ref)

## Model evaluation
- [`full_model`](@ref)

## Fitting
- [`PriorPair`](@ref), [`poisson_ll`](@ref), [`build_prior`](@ref), 
  [`build_posterior`](@ref)

## Utils
- [`cut_data`](@ref), [`get_peak_features`](@ref)

## Plotting
- [`plot_data`](@ref)

# Quick start

```julia
using GammaPeakFits

# Configurations
configs = FitConfigs(
              mu = 2048.0,  # keV
              sigma = 5.0,  # keV
          )

# Generate data
A = 1000.0              # counts
C_const = 100.0         # counts/keV
lower_limit = 1.0       # keV
upper_limit = 4096.0    # keV
bin_size = 0.5          # keV

gaussian_params = GaussianParams(A = A, mu = configs.mu, sigma = configs.sigma)
peak_params = PeakParams(gaussian = gaussian_params)
constPoly_params = ConstPolyParams(C = C_const)
background_params = BackgroundParams(constPoly = constPoly_params)
generation_modelParams = ModelParams(peak = peak_params, background = background_params)
data = SpectrumData(lower_limit, upper_limit, bin_size, generation_modelParams)

# or use existing data instead
# data = SpectrumData(
#            loaded_binCenters,  # keV
#            loaded_weights,     # counts/bin
#        )

# cut appropriate fit window
window_size = 100.0 # keV
fit_data = cut_data(data, configs.mu, window_size)

# Specify which components to include for fitting
peak_params = PeakParams(gaussian = Enabled())
background_params = BackgroundParams(constPoly = Enabled())
fit_modelParams = ModelParams(peak = peak_params, background = background_params)

# Build the prior
prior = build_prior(fit_data, fit_modelParams, configs)

# Build the posterior
posterior = build_posterior(fit_data, prior, configs)

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
using DensityInterface: logfuncdensity
using Distributions
using QuadGK: quadgk
using SpecialFunctions: erf, erfc, logerfcx
using ValueShapes: NamedTupleDist

# Component/container parameter structs and Enabled/Disabled presence markers
include("params.jl")
# SpectrumData container, marker-based and synthetic-data constructors
include("data.jl")
# FitConfigs, and AbstractIntegrationMethod subtypes
include("configs.jl")
# Component model evaluation (gaussian, compton, ...) and combined models
include("models.jl")
# Poisson log-likelihood, prior and posterior construction
include("fitting.jl")
# Analytical, numerical (quadgk) and midpoint bin integrals
include("integrals.jl")
# Data slicing (cut_data), peak-feature estimation, is_present
include("utils.jl")
# Plotting
include("plotting.jl")

# Types - presence markers
export AbstractComponent
export Enabled
export Disabled

# Types - Integration methods
export AbstractIntegrationMethod
export Analytical
export Numerical
export Midpoint

# Parameter structs - components
export GaussianParams
export ComptonParams
export ExGaussianParams
export QuadPolyParams
export LinPolyParams
export ConstPolyParams

# Parameter structs - containers
export PeakParams
export BackgroundParams
export ModelParams

# Data structs
export SpectrumData

# Configuration struct
export FitConfigs

# Model evaluation
export full_model

# Fitting
export PriorPair
export poisson_ll
export build_prior
export build_posterior

# Plotting
export plot_data

# Utils
export cut_data
export get_peak_features

end
