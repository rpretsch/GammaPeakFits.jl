"""
Module for Bayesian fitting of gamma-ray peaks in a pre-binned energy spectrum.

Provides parametric models for gamma-peak shapes (Gaussian core, Compton edge, ex-Gaussian 
tails) and polynomial backgrounds, along with utilities to construct `BAT.jl` priors and 
posteriors.

!!! note "Work in progress":
The ex-Gaussian tail functionality is currently incomplete. The [`exGaussian`](@ref)
evaluation function exists, but tail components are not yet wired into 
[`build_prior`](@ref) or [`build_posterior`](@ref).

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
mu = 2048.0
sigma = 10.0

# Specify model components
peak = PeakParams(gaussian = true, compton = true)
bg = BackgroundParams(constPoly = true)
model = ModelParams(peak = peak, background = bg)

# Build prior and posterior
prior = build_prior(model, mu, sigma, peak_height = 100.0, peak_area = 1000.0)
data = SpectrumData(bin_centers = 1.0:4096.0, weights = counts, bin_size = 1.0)
posterior = build_posterior(data, prior)
```
"""
module GammaPeakFits

using BAT
using CairoMakie
using Distributions
using QuadGK: quadgk
using SpecialFunctions: erfc
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
