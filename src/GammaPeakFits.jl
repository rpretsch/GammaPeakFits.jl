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
  [`AbstractExGaussianParams`](@ref), [`LowETailParams`](@ref),
  [`HighETailParams`](@ref), [`QuadPolyParams`](@ref), [`LinPolyParams`](@ref),
  [`ConstPolyParams`](@ref)
- Containers: [`PeakParams`](@ref), [`BackgroundParams`](@ref), [`ModelParams`](@ref)
- Data: [`SpectrumData`](@ref)
- Configuration: [`FitConfigs`](@ref)

## Data
- [`bin_energies`](@ref), [`rebin`](@ref), [`sample_spectrum`](@ref)

## Model evaluation
- [`full_model`](@ref)

## Fitting
- [`PriorPair`](@ref), [`poisson_ll`](@ref), [`build_prior`](@ref), 
  [`build_posterior`](@ref)

## Utils
- [`cut_data`](@ref)

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
energy_range = (0.0, 4096.0)  # keV, outer bin edges
bin_size = 0.5          # keV

gaussian_params = GaussianParams(A = A, mu = configs.mu, sigma = configs.sigma)
peak_params = PeakParams(gaussian = gaussian_params)
constPoly_params = ConstPolyParams(C = C_const)
background_params = BackgroundParams(constPoly = constPoly_params)
generation_modelParams = ModelParams(peak = peak_params, background = background_params)
data = sample_spectrum(generation_modelParams, bin_size; energy_range)

# or use existing data instead
# data = SpectrumData(
#            loaded_binCenters,  # keV
#            loaded_weights,     # counts/bin
#        )

# cut appropriate fit window
fit_data = cut_data(data, configs)

# Specify which components to include for fitting
peak_params = PeakParams(gaussian = Enabled())
background_params = BackgroundParams(constPoly = Enabled())
fit_modelParams = ModelParams(peak = peak_params, background = background_params)

# Build the prior
prior = build_prior(fit_data, fit_modelParams, configs)

# Build the posterior
posterior = build_posterior(fit_data, prior, configs)

# Sample with BAT.jl
# using BAT: bat_sample
# using StatsBase: mean
#
# result = bat_sample(
#              posterior, 
#              TransformedMCMC(proposal=RandomWalk(), nsteps=10^5, nchains=4)
#          ).result
#
# mean_result = mean(result)
# mean_params = ModelParams(mean_result)
```
"""
module GammaPeakFits

using BAT
using Distributions
import LegendSpecFits
using LegendSpecFits:
    GaussPeak,
    LowETailPeak,
    HighETailPeak,
    GaussianStep,
    PolynomialShape,
    binned_poisson_likelihood
using MeasureBase: AbstractMeasure
using PropertyFunctions: @pf, PPath, PropertyFunction, input_property_paths
using StatsBase: Histogram, fit, mean, midpoints
using ValueShapes: NamedTupleDist

# Spectrum check, binning and synthetic spectra
include("data.jl")
# Expected counts of measure models
include("models.jl")
# Prior and posterior construction for @pf model kernels
include("fitting.jl")
# Data slicing (cut_data), peak-feature estimation
include("utils.jl")

# Data
export bin_energies
export rebin
export sample_spectrum

# Model evaluation
export expected_counts

# Fitting
export peak_priors
export background_priors
export build_prior
export build_posterior

# Utils
export cut_data

# Re-exports - spectral shapes (LegendSpecFits) and model kernels (PropertyFunctions)
export GaussPeak
export LowETailPeak
export HighETailPeak
export GaussianStep
export PolynomialShape
export @pf

end
