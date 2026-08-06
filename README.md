# GammaPeakFits

| Documentation | Build Status |
|:-------------:|:------------:|
| [![Stable Docs][docs-stable-img]][docs-stable-url] [![Dev Docs][docs-dev-img]][docs-dev-url] [![Julia Version][julia-img]][julia-url] [![License][license-img]](LICENSE) | [![Build Status][CI-img]][CI-url] [![Coverage][Cov-img]][Cov-url] [![Aqua QA][aqua-img]][aqua-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://rpretsch.github.io/GammaPeakFits.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://rpretsch.github.io/GammaPeakFits.jl/dev/

[julia-img]: https://img.shields.io/badge/julia-%3E%3D1.10-blue
[julia-url]: https://julialang.org

[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg

[CI-img]: https://github.com/rpretsch/GammaPeakFits.jl/actions/workflows/CI.yml/badge.svg?branch=main
[CI-url]: https://github.com/rpretsch/GammaPeakFits.jl/actions/workflows/CI.yml?query=branch%3Amain

[Cov-img]: https://codecov.io/gh/rpretsch/GammaPeakFits/branch/main/graph/badge.svg
[Cov-url]: https://codecov.io/gh/rpretsch/GammaPeakFits

[aqua-img]: https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

Bayesian gamma-ray peak fitting on a binned energy spectrum with a composable
shape model and Poisson likelihood, built on
[BAT.jl](https://github.com/bat/BAT.jl).

## File Structure

```text
GammaPeakFits/
├── Project.toml            Package metadata and dependencies
├── src/
│   ├── GammaPeakFits.jl    Main module: imports, exports, and includes
│   ├── types.jl            Parameter structs for model components and containers
│   ├── models.jl           Model evaluation functions
│   ├── fitting.jl          Poisson likelihood, prior and posterior construction
│   └── utils.jl            Data slicing, plotting and peak-feature extraction
├── test/
│   ├── runtests.jl         Top-level test runner
│   ├── test_types.jl       Struct construction and field default tests
│   ├── test_models.jl      Model evaluation and Bool sentinel guard tests
│   ├── test_fitting.jl     Prior, posterior, and likelihood tests
│   └── test_utils.jl       Data slicing and peak-feature tests
└── README.md
```

## Installation

This package is unregistered. Install it by cloning the repository and adding
it with Julia's package manager:

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

## Model Components

Each component can be enabled or disabled:

- Set it to `false` (default): the component is excluded.
- Set it to `true`: the component is included in the fit using its prior.

The entire peak or background can be disabled by setting the corresponding
field in `ModelParams` to `nothing`.

### Peak

#### Gaussian Core

```math
f(x) = \frac{A}{\sqrt{2\pi}\sigma}\, 
       \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
       
```

| Parameter | Unit | Description |
| --- | --- | --- |
| `A` | counts | Total integrated peak area |
| `mu` | keV | Centroid position |
| `sigma` | keV | Standard deviation |

#### Compton-Edge Step

```math
f(x) = \frac{h}{2}\,
       \text{erfc}\!\left(\frac{x-\mu}{\sqrt{2}\sigma}\right)
```

| Parameter | Unit | Description |
| --- | --- | --- |
| `h` | counts/keV | Step height |
| `mu` | keV | Centroid position |
| `sigma` | keV | Standard deviation |

#### Ex-Gaussian Tails

The ex-Gaussian component models asymmetric peak tailing (low- or high-energy):

```math
f(x) = \frac{A}{2\tau}\,
       \exp\!\left(\frac{1}{2}\left(\frac{\sigma}{\tau}\right)^2
       \pm\frac{x-\mu}{\tau}\right)\,
       \text{erfc}\!\left(\frac{1}{\sqrt{2}}\left(\frac{\sigma}{\tau}
       \pm\frac{x-\mu}{\sigma}\right)\right)
```

| Parameter | Unit | Description |
| --- | --- | --- |
| `A` | counts | Total integrated tail area |
| `tau` | keV | Exponent relaxation time of the exponential tail |
| `is_lowEnergyTail` | Boolean | Tail direction (`true`/`false` for low-/high-energy tails, respectively) |
| `mu` | keV | Centroid position of the gaussian |
| `sigma` | keV | Standard deviation of the gaussian |

The low-/high-energy tails correspond to a $+$ / $-$ sign for the $\pm$ sign
above, respectively.

For numerical stability this is evaluated in log-space as

```math
\log f(x) = \log A - \log(2\tau) 
            -\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2 
            +\text{logerfcx}\!\left(\frac{1}{\sqrt{2}}\left(\frac{\sigma}{\tau}
            \pm\frac{x-\mu}{\sigma}\right)\right)
```

using `SpecialFunctions.logerfcx`.

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
            peak_area = peak_area,
       )

# Build the posterior
posterior = build_posterior(fit_data, prior)

# Sample with BAT.jl
# result = bat_sample(
#              posterior, 
#              TransformedMCMC(proposal=RandomWalk(), nsteps=10^5, nchains=4)
#          )
```

## License

MIT — see [LICENSE](LICENSE).
