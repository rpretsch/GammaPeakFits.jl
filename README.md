# GammaPeakFits

|                                                                              Documentation                                                                               |                                            Build Status                                            |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| [![Stable Docs][docs-stable-img]][docs-stable-url] [![Dev Docs][docs-dev-img]][docs-dev-url] [![Julia Version][julia-img]][julia-url] [![License][license-img]](LICENSE) | [![Build Status][CI-img]][CI-url] [![Coverage][Cov-img]][Cov-url] [![Aqua QA][aqua-img]][aqua-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://rpretsch.github.io/GammaPeakFits.jl/stable/
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://rpretsch.github.io/GammaPeakFits.jl/dev/
[julia-img]: https://img.shields.io/badge/julia-%3E%3D1.11-blue
[julia-url]: https://julialang.org
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg
[CI-img]:
  https://github.com/rpretsch/GammaPeakFits.jl/actions/workflows/CI.yml/badge.svg?branch=main
[CI-url]:
  https://github.com/rpretsch/GammaPeakFits.jl/actions/workflows/CI.yml?query=branch%3Amain
[Cov-img]:
  https://codecov.io/gh/rpretsch/GammaPeakFits.jl/branch/main/graph/badge.svg
[Cov-url]: https://codecov.io/gh/rpretsch/GammaPeakFits.jl
[aqua-img]: https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

Bayesian gamma-ray peak fitting on binned energy spectra: models are kernels
that map parameters to a spectral measure, e.g. composed from the shapes of
[LegendSpecFits.jl](https://github.com/legend-exp/LegendSpecFits.jl), fitted
with a binned Poisson likelihood and sampled with
[BAT.jl](https://github.com/bat/BAT.jl).

## File Structure

```text
GammaPeakFits/
├── Project.toml   Package metadata and dependencies
├── docs/          Documenter.jl docs site
├── src/           Module sources (entry point: src/GammaPeakFits.jl)
├── test/          Test suite (entry point: test/runtests.jl)
└── README.md
```

## Installation

GammaPeakFits depends on LegendSpecFits.jl from the
[LegendJuliaRegistry](https://github.com/legend-exp/LegendJuliaRegistry); add
that registry once:

```shell
julia -e 'using Pkg; Pkg.Registry.add(url="https://github.com/legend-exp/LegendJuliaRegistry.git")'
```

This package is unregistered. Install it by cloning the repository and adding it
with Julia's package manager:

### For users

```shell
julia -e 'using Pkg; Pkg.add(url="https://github.com/rpretsch/GammaPeakFits.jl.git")'
```

### For development

```shell
julia -e 'using Pkg; Pkg.develop(url="https://github.com/rpretsch/GammaPeakFits.jl.git")'
```

Or if already cloned locally:

```shell
julia -e 'using Pkg; Pkg.develop(path="/path/to/GammaPeakFits")'
```

## Models

A model is a `@pf` kernel that maps parameters (`$name`) to a spectral measure
in counts/keV. Any measure works; the re-exported shapes below can be combined
with `+` and `*`:

| Shape                             | Density                                                                  |
| --------------------------------- | ------------------------------------------------------------------------ |
| `GaussPeak(mu, sigma)`            | Gaussian peak with unit area                                             |
| `LowETailPeak(mu, sigma, theta)`  | ex-Gaussian peak with unit area, tail of scale `theta` to lower energies |
| `HighETailPeak(mu, sigma, theta)` | same with the tail to higher energies                                    |
| `GaussianStep(mu, sigma)`         | smoothed step of height 1 below `mu` (Compton edge)                      |
| `PolynomialShape(b, b1, b2; x0)`  | `b + b1 (x - x0) + b2 (x - x0)^2`                                        |

Bin masses of these shapes are exact (differences of their cumulative
functions); measures without one are integrated per bin with Simpson's rule,
which can miss peaks that are narrow compared to the bins.

A peak with a low-energy tail on a linear background:

```julia
model = @pf $n * ((1 - $alpha) * GaussPeak($mu, $sigma) +
                  $alpha * LowETailPeak($mu, $sigma, $theta)) +
            PolynomialShape($b, $b1; x0 = 2048.0)
```

`n` is the number of peak counts and `alpha` the tail fraction. Parameters can
be grouped, e.g. `$peak.n`, `$peak.mu`; the prior is then nested in the same
way. `expected_counts(model(params), hist)` gives the expected counts per bin,
e.g. for plotting a fit.

## Priors

`peak_priors(hist; mu, sigma)` and `background_priors(hist; mu, sigma)` return
`NamedTuple`s of weakly informative default priors estimated from the data:

| Name    | Default prior                                | Block               |
| ------- | -------------------------------------------- | ------------------- |
| `n`     | `Uniform(0, 3 * peak_area)`                  | `peak_priors`       |
| `alpha` | `Uniform(0, 1)`                              | `peak_priors`       |
| `mu`    | `Normal(mu, 0.6)`                            | `peak_priors`       |
| `sigma` | `truncated(Normal(sigma, 0.6), eps(), Inf)`  | `peak_priors`       |
| `theta` | `Uniform(eps(), 5 * sigma)`                  | `peak_priors`       |
| `b`     | `Uniform(0, 2 * level)`                      | `background_priors` |
| `b1`    | `Uniform(-2 * level / h, 2 * level / h)`     | `background_priors` |
| `b2`    | `Uniform(-2 * level / h^2, 2 * level / h^2)` | `background_priors` |
| `h`     | `Uniform(0, 4 * level)`                      | `background_priors` |

`peak_area` is estimated from the highest count rate within `mu ± 3 sigma` above
the mean background `level` (counts/keV) outside that region; both are floored
at one count per bin. `h` is the half-width of the histogram's range, so the
`b1`/`b2` bounds assume the polynomial is expanded around the window center
(`x0`).

`build_prior(model, defaults)` takes the entry of `defaults` for each parameter
of the kernel and ignores the rest; replace single entries with `merge`:

```julia
using Distributions: Uniform

defaults = merge(
    peak_priors(fit_data; mu, sigma),
    background_priors(fit_data; mu, sigma),
    (n = Uniform(0, 1e4),),
)
prior = build_prior(model, defaults)
```

For grouped parameters, `defaults` is nested the same way, e.g.
`(peak = peak_priors(...), bkg = background_priors(...))`.

## Usage

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
model = @pf $n * GaussPeak($mu, $sigma) + PolynomialShape($b)

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

## License

MIT — see [LICENSE](LICENSE).
