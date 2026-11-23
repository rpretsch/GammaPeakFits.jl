"""
Bayesian fitting of gamma-ray peaks in binned energy spectra.

Data are one-dimensional `StatsBase.Histogram`s of counts with uniform bins. Models are
`@pf` kernels `params -> measure`, e.g. combined from the re-exported LegendSpecFits
shapes with `+` and `*`; the likelihood is LegendSpecFits' binned Poisson likelihood,
sampled with `BAT.jl`.

# Exports

## Data
- [`bin_energies`](@ref), [`rebin`](@ref), [`cut_data`](@ref), [`sample_spectrum`](@ref)

## Models
- Re-exported: `GaussPeak`, `LowETailPeak`, `HighETailPeak`, `GaussianStep`,
  `PolynomialShape` (LegendSpecFits) and `@pf` (PropertyFunctions)
- [`expected_counts`](@ref)

## Fitting
- [`peak_priors`](@ref), [`background_priors`](@ref), [`build_prior`](@ref),
  [`build_posterior`](@ref)

# Quick start

```julia
using GammaPeakFits

# Expected peak
mu = 2048.0     # keV
sigma = 5.0     # keV

# Generate data, or use a measured spectrum, e.g. `bin_energies(energies, 0.5)`
peak_counts = 1000.0          # counts
background = 100.0            # counts/keV
energy_range = (0.0, 4096.0)  # keV, outer bin edges
bin_size = 0.5                # keV

truth = peak_counts * GaussPeak(mu, sigma) + PolynomialShape(background)
data = sample_spectrum(truth; bin_size, energy_range)

# cut appropriate fit window
fit_data = cut_data(data; center = mu, width = 10 * sigma)

# Model kernel: parameters -> spectral model
model = @pf \$n * GaussPeak(\$mu, \$sigma) + PolynomialShape(\$b)

# Data-driven default priors, the prior takes the entries the model uses
defaults = merge(peak_priors(fit_data; mu, sigma), background_priors(fit_data; mu, sigma))
prior = build_prior(model, defaults)
posterior = build_posterior(fit_data, model, prior)

# Sample with BAT.jl
# using BAT: bat_sample, TransformedMCMC, RandomWalk
# using StatsBase: mean
#
# samples = bat_sample(
#               posterior,
#               TransformedMCMC(proposal = RandomWalk(), nsteps = 10^5, nchains = 4),
#           ).result
#
# # Expected counts of the fitted model, e.g. for plotting
# fitted = expected_counts(model(mean(samples)), fit_data)
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
