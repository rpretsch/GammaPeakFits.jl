# GammaPeakFits

[![Build Status](https://github.com/rpretsch/GammaPeakFits.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rpretsch/GammaPeakFits.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Julia Version](https://img.shields.io/badge/julia-%3E%3D1.10-blue)](https://julialang.org)

Bayesian gamma-ray peak fitting on a binned energy spectrum with a composable shape model and Poisson likelihood using [BAT.jl](https://github.com/bat/BAT.jl).

## File Structure

| File | Description |
| --- | --- |
| `Project.toml` | Package metadata and dependencies |
| `src/GammaPeakFits.jl` | Main module: imports and includes |
| `src/types.jl` | Parameter structs for all model components |
| `src/models.jl` | Model evaluation: Gaussian, Compton, tails, background, peak shape |
| `src/likelihoods.jl` | Poisson log-likelihood with bin-integrated expected counts |
| `test/runtests.jl` | Test runner entry point |

## Model Components

### Gaussian Core

```math
f(x) = \frac{A}{\sigma\sqrt{2\pi}} \,
       \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
```

### Compton-Edge Step

```math
f(x) = \frac{h}{2} \,
       \text{erfc}\!\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)
```

### Ex-Gaussian Tail

```math
f(x) = \frac{A\lambda}{2} \,
       \exp\!\left(\frac{\lambda}{2}
       (2\mu + \lambda\sigma^2 - 2x)\right) \,
       \text{erfc}\!\left(
       \frac{\mu + \lambda\sigma^2 - x}{\sigma\sqrt{2}}\right)
```

Sign of `lambda` determines the tail direction: `lambda > 0` produces a high-energy tail, `lambda < 0` produces a low-energy tail.

### Background

```math
f(x) = b_2 (x - \mu)^2 + b_1 (x - \mu) + b_0
```

Each polynomial coefficient is optional — set to `nothing` to exclude the term.

## License

MIT — see [LICENSE](LICENSE).
