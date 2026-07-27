"""
    poisson_ll(data::SpectrumData, params::ModelParams)

Compute the Poisson log-likelihood for the model given observed counts.

The expected count in each bin is obtained by integrating [`full_model`](@ref) over the bin 
width `bin_size` using `QuadGK.jl`.
The log-probability of the observed integer count `weight` is then evaluated under a
Poisson distribution with that expected rate.

# Arguments
- `data::SpectrumData`: data container
- `params::ModelParams`: model parameters

# Returns
sum of log-likelihoods across bins (total log-likelihood)

# See also
- [`full_model`](@ref) for the underlying model
- [`ModelParams`](@ref) for the parameter structure
"""
function poisson_ll(data::SpectrumData, params::ModelParams)
    expected_counts = quadgk.(
        x -> full_model(x, Ref(params)),
        data.bin_centers - data.bin_size/2,
        data.bin_centers + data.bin_size/2,
    )[1]
    result_vector = logpdf.(Poisson(expected_counts), data.weights)
    return sum(result_vector)
end

"""
    build_prior(
        params::ModelParams,
        mu::Real,
        sigma::Real,
        peak_height::Real,
        peak_area::Real,
    )

Construct a prior distribution over the model parameters for Bayesian fitting.

Components that are `false` (or `nothing` for container fields) are skipped. Components set 
to `true` or to a concrete parameter struct receive a weakly informative prior.

# Arguments
- `params::ModelParams`: model specification indicating which components are enabled
- `mu::Real`: expected centroid position of the peak in keV
- `sigma::Real`: expected standard deviation of the Gaussian core in keV
- `peak_height::Real`: approximate peak height in counts/keV
- `peak_area::Real`: approximate integrated peak area in counts

# Returns
A `NamedTupleDist` (via `distprod`) over the enabled component parameters.

# Throws
`ArgumentError` if both `params.peak` and `params.background` are `nothing`.

# Details

The following priors are defined per enabled component:

| Symbol | Prior | Component |
| --- | --- | --- |
| `:mu` | `Normal(mu, 0.6)` | all |
| `:sigma` | `truncated(Normal(sigma, 0.6), 0, Inf)` | `peak.gaussian`, `peak.lowEnergyTail`, `peak.highEnergyTail` |
| `:gaussian_A` | `Uniform(0, peak_area)` | `peak.gaussian` |
| `:compton_h` | `Uniform(0, peak_height)` | `peak.compton` |
| `:lowEnergyTail_A` | TODO | `peak.lowEnergyTail` |
| `:lowEnergyTail_lambda` | TODO | `peak.lowEnergyTail` |
| `:highEnergyTail_A` | TODO | `peak.highEnergyTail` |
| `:highEnergyTail_lambda` | TODO | `peak.highEnergyTail` |
| `:quadPoly_C` | `Uniform(-1, 1)` | `background.quadPoly` |
| `:linPoly_C` | `Uniform(-10, 10)` | `background.linPoly` |
| `:constPoly_C` | `Uniform(0, peak_height)` | `background.constPoly` |

# Examples

```julia
mu = 2048.0
sigma = 10.0
p = PeakParams(gaussian = true, compton = true)
b = BackgroundParams(constPoly = true)
m = ModelParams(peak = p, background = b)
prior = build_prior(m, mu, sigma, 100.0, 1000.0)
```

# See also
- [`ModelParams`](@ref), [`PeakParams`](@ref), [`BackgroundParams`](@ref) for the model
  specification
- [`poisson_ll`](@ref) for the likelihood that uses these priors
"""
function build_prior(
    params::ModelParams,
    mu::Real,
    sigma::Real,
    peak_height::Real,
    peak_area::Real,
)
    if (isnothing(params.peak) && isnothing(params.background))
        throw(ArgumentError("No model specified"))
    end

    priors = Dict()
    priors[:mu] = Normal(mu, 0.6)
    if !isnothing(params.peak) && (
        params.peak.gaussian !== false ||
        params.peak.lowEnergyTail !== false ||
        params.peak.highEnergyTail !== false
    )
        priors[:sigma] = truncated(Normal(sigma, 0.6), 0, Inf)
    end

    if !isnothing(params.peak)
        params.peak.gaussian !== false && (priors[:gaussian_A] = Uniform(0, peak_area))
        params.peak.compton !== false && (priors[:compton_h] = Uniform(0, peak_height))
        if params.peak.lowEnergyTail !== false
            # TODO
        end
        if params.peak.highEnergyTail !== false
            # TODO
        end
    end

    if !isnothing(params.background)
        params.background.quadPoly !== false && (priors[:quadPoly_C] = Uniform(-1, 1))
        params.background.linPoly !== false && (priors[:linPoly_C] = Uniform(-10, 10))
        params.background.constPoly !== false &&
            (priors[:constPoly_C] = Uniform(0, peak_height))
    end

    return distprod(; priors...)
end
