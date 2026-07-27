# GammaPeakFits

[![Build Status](https://github.com/rpretsch/GammaPeakFits.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rpretsch/GammaPeakFits.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Julia Version](https://img.shields.io/badge/julia-%3E%3D1.10-blue)](https://julialang.org)

Bayesian gamma-ray peak fitting on a binned energy spectrum with a composable shape model and Poisson likelihood, built on
[BAT.jl](https://github.com/bat/BAT.jl).

## File Structure

```text
GammaPeakFits/
├── Project.toml            Package metadata and dependencies
├── src/
│   ├── GammaPeakFits.jl    Main module: imports, exports, and includes
│   ├── types.jl            Parameter structs for model components and containers
│   ├── models.jl           Model evaluation functions
│   └── fitting.jl          Poisson likelihood, prior and posterior construction
├── test/
│   ├── runtests.jl         Top-level test runner
│   ├── test_types.jl       Struct construction and field default tests
│   ├── test_models.jl      Model evaluation and Bool sentinel guard tests
│   └── test_fitting.jl     Prior, posterior, and likelihood tests
└── README.md
```

## Model Components

Each component can be enabled or disabled:

- Set it to `false` (default): the component is excluded.
- Set it to `true`: the component is included in the fit using its prior.

The entire peak or background can be disabled by setting the corresponding field in [`ModelParams`](@ref) to `nothing`.

### Peak

#### Gaussian Core

```math
f(x) = \frac{A}{\sigma\sqrt{2\pi}} \,
       \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
```

| Parameter | Unit | Description |
| --- | --- | --- |
| `A` | counts | Total integrated peak area |
| `mu` | keV | Centroid position |
| `sigma` | keV | Standard deviation |

#### Compton-Edge Step

```math
f(x) = \frac{h}{2} \,
       \text{erfc}\!\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)
```

| Parameter | Unit | Description |
| --- | --- | --- |
| `h` | counts/keV | Step height |
| `mu` | keV | Centroid position |
| `sigma` | keV | Standard deviation |

#### Ex-Gaussian Tails

> **WIP**: Tail evaluation is implemented, but tail priors and posterior reconstruction are > not yet wired into `build_prior` /
> `build_posterior`.

The ex-Gaussian component models asymmetric peak tailing (low- or high-energy):

```math
f(x) = \frac{A\lambda}{2} \,
       \exp\!\left(\frac{\lambda}{2}
       (2\mu + \lambda\sigma^2 - 2x)\right) \,
       \text{erfc}\!\left(
       \frac{\mu + \lambda\sigma^2 - x}{\sigma\sqrt{2}}\right)
```

| Parameter | Unit | Description |
| --- | --- | --- |
| `A` | counts | Total integrated tail area |
| `lambda` | 1/keV | Rate parameter of the exponential tail |
| `mu` | keV | Centroid position |
| `sigma` | keV | Standard deviation |

### Background

#### Quadratic Polynomial Term

```math
f(x) = C \cdot (x - \mu)^2
```

| Parameter | Unit | Description |
| --- | --- | --- |
| `C` | counts/keV³ | Scaling coefficient |
| `mu` | keV | Centring value (usually the peak centroid) |

#### Linear Polynomial Term

```math
f(x) = C \cdot (x - \mu)
```

| Parameter | Unit | Description |
| --- | --- | --- |
| `C` | counts/keV² | Scaling coefficient |
| `mu` | keV | Centring value (usually the peak centroid) |

#### Constant Polynomial Term

```math
f(x) = C
```

| Parameter | Unit | Description |
| --- | --- | --- |
| `C` | counts/keV | Constant offset |

## Usage

```julia
using GammaPeakFits

mu = 2048.0
sigma = 10.0

# Specify which components to include
p = PeakParams(gaussian = true, compton = true)
b = BackgroundParams(linPoly = true)
model = ModelParams(peak = p, background = b)

# Build the prior
prior = build_prior(
            model, mu, sigma,
            peak_height = 100.0,
            peak_area = 1000.0,
        )

# Build the posterior
data = SpectrumData(bin_centers = 1.0:4096.0, weights = counts, bin_size = 1.0)
posterior = build_posterior(data, prior)

# Sample with BAT.jl
# result = bat_sample(post, MCMCSerial(64, nsteps = 100_000))
```

## License

MIT — see [LICENSE](LICENSE).
