"""
    PriorPair

Shorthand for `Pair{Symbol,Distribution}``, used in [`build_prior`](@ref).

# See also 
- [`build_prior`](@ref)
"""
const PriorPair = Pair{Symbol,Distribution}

"""
    poisson_ll(data::SpectrumData, params::ModelParams, configs::FitConfigs)

Compute the Poisson log-likelihood for the model given observed counts.

The expected count in each bin is obtained by integrating [`full_model`](@ref) over the bin 
width `bin_size`.
The log-probability of the observed integer count `weight` is then evaluated under a
Poisson distribution with that expected rate.

# Arguments
- `data::SpectrumData`: data container
- `params::ModelParams`: model parameters
- `configs::FitConfigs`: Fitting configurations

# Returns
- `-Inf` if any expected counts are negative or non-finite (unphysical model configuration)
- sum of log-likelihoods across bins (total log-likelihood) otherwise

# See also
- [`full_model`](@ref) for the underlying model
- [`ModelParams`](@ref) for the parameter structure
- [`AbstractIntegrationMethod`](@ref) for the integration method selection
"""
function poisson_ll(data::SpectrumData, params::ModelParams, configs::FitConfigs)
    expected_counts = _expected_counts(configs.integration_method, data, params)
    any(x -> (x < 0 || !isfinite(x)), expected_counts) && return -Inf
    result_vector = logpdf.(Poisson.(expected_counts), data.weights)
    return sum(result_vector)
end

"""
    _expected_counts(method::AbstractIntegrationMethod, data::SpectrumData, params::ModelParams)

Compute the expected counts per bin by integrating the full model according to the 
integration method, dispatching on `method`.

# Arguments
- `method::AbstractIntegrationMethod`: the integration method to use
- `data::SpectrumData`: binned spectrum data
- `params::ModelParams`: model parameters

# Returns
- An array of expected counts per bin

# See also
- [`AbstractIntegrationMethod`](@ref) for the available integration methods
- [`poisson_ll`](@ref) for the likelihood that consumes the result
- [`analytical_integral`](@ref), [`numerical_integral`](@ref), [`midpoint_integral`](@ref)
  for the underlying integrals
"""
_expected_counts(::Analytical, data::SpectrumData, params::ModelParams) =
    analytical_integral(data, params)
_expected_counts(::Numerical, data::SpectrumData, params::ModelParams) =
    numerical_integral(data, params)
_expected_counts(::Midpoint, data::SpectrumData, params::ModelParams) =
    midpoint_integral(data, params)

"""
    build_prior(
        data::SpectrumData, 
        params::ModelParams, 
        configs::FitConfigs;
        priors::Vector{PriorPair} = PriorPair[],
    )

Construct a prior distribution over the model parameters for Bayesian fitting.

Components that are `Disabled()` are skipped. Components set to `Enabled()` or to a 
concrete parameter struct receive a weakly informative prior.

The expected peak centroid `configs.mu` and core width `configs.sigma` are taken from 
[`FitConfigs`](@ref) and can be tuned there.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `model::ModelParams`: model specification indicating which components are enabled
- `configs::FitConfigs`: fitting configuration

# Keyword arguments
- `priors::PriorPair`: optional custom priors that overwrite the here set 
  defaults for the specified symbols. Default: `PriorPair[]`

# Returns
- A `NamedTupleDist` (via `distprod`) over the enabled component parameters

# Throws
- An `ArgumentError` if both `model.peak` and `model.background` are `Disabled()`
- An `ArgumentError` if a present container holds no enabled components 
  (e.g. `PeakParams()`)
-  An `ArgumentError` if the specified `priors` do not fit the used model

# Details

The following priors are defined per enabled component:

| Symbol | Prior | Component |
| --- | --- | --- |
| `:mu` | `Normal(configs.mu, 0.6)` | all except `background.constPoly` |
| `:sigma` | `truncated(Normal(configs.sigma, 0.6), eps(), Inf)` | all `peak` components |
| `:gaussian_A` | `Uniform(0, 2 * peak_area)` | `peak.gaussian` |
| `:compton_h` | `Uniform(0, 4 * mean_background)` | `peak.compton` |
| `:lowEnergyTail_A` | `Uniform(eps(), peak_area)` | `peak.lowEnergyTail` |
| `:lowEnergyTail_tau` | `Uniform(eps(), 10)` | `peak.lowEnergyTail` |
| `:highEnergyTail_A` | `Uniform(eps(), peak_area)` | `peak.highEnergyTail` |
| `:highEnergyTail_tau` | `Uniform(eps(), 10)` | `peak.highEnergyTail` |
| `:quadPoly_C` | `Uniform(-1, 1)` | `background.quadPoly` |
| `:linPoly_C` | `Uniform(-10, 10)` | `background.linPoly` |
| `:constPoly_C` | `Uniform(0, 2 * mean_background)` | `background.constPoly` |

# See also
- [`Enabled`](@ref), [`Disabled`](@ref), [`AbstractComponent`](@ref) for the component 
  management
- [`FitConfigs`](@ref) for tuning the `mu` and `sigma` prior centers
- [`ModelParams`](@ref), [`PeakParams`](@ref), [`BackgroundParams`](@ref) for the model
  specification
- [`poisson_ll`](@ref) for the likelihood that uses these priors
"""
function build_prior(
    data::SpectrumData,
    model::ModelParams,
    configs::FitConfigs;
    priors::Vector{PriorPair} = PriorPair[],
)
    _, peak_area, mean_background = get_peak_features(data, configs.mu, configs.sigma)

    peak_model = model.peak
    background_model = model.background

    if (!is_present(peak_model) && !is_present(background_model))
        throw(ArgumentError("No model specified"))
    end

    default_priors = PriorPair[]

    needs_mu =
        is_present(peak_model) && (
            is_present(peak_model.gaussian) ||
            is_present(peak_model.lowEnergyTail) ||
            is_present(peak_model.highEnergyTail) ||
            is_present(peak_model.compton)
        ) ||
        is_present(background_model) &&
        (is_present(background_model.quadPoly) || is_present(background_model.linPoly))
    needs_mu && push!(default_priors, :mu => Normal(configs.mu, 0.6))

    needs_sigma =
        is_present(peak_model) && (
            is_present(peak_model.gaussian) ||
            is_present(peak_model.lowEnergyTail) ||
            is_present(peak_model.highEnergyTail) ||
            is_present(peak_model.compton)
        )
    needs_sigma &&
        push!(default_priors, :sigma => truncated(Normal(configs.sigma, 0.6), eps(), Inf))

    if is_present(peak_model)

        is_present(peak_model.gaussian) &&
            push!(default_priors, :gaussian_A => Uniform(0, 2 * peak_area))

        is_present(peak_model.compton) &&
            push!(default_priors, :compton_h => Uniform(0, 4 * mean_background))

        if is_present(peak_model.lowEnergyTail)
            push!(default_priors, :lowEnergyTail_A => Uniform(eps(), peak_area))
            push!(default_priors, :lowEnergyTail_tau => Uniform(eps(), 10))
        end

        if is_present(peak_model.highEnergyTail)
            push!(default_priors, :highEnergyTail_A => Uniform(eps(), peak_area))
            push!(default_priors, :highEnergyTail_tau => Uniform(eps(), 10))
        end

    end

    if is_present(background_model)

        is_present(background_model.quadPoly) &&
            push!(default_priors, :quadPoly_C => Uniform(-1, 1))

        is_present(background_model.linPoly) &&
            push!(default_priors, :linPoly_C => Uniform(-10, 10))

        is_present(background_model.constPoly) &&
            push!(default_priors, :constPoly_C => Uniform(0, 2 * mean_background))

    end

    isempty(default_priors) && throw(ArgumentError("No model specified"))

    if !isempty(priors)
        default_symbols = Set(first.(default_priors))
        user_symbols = Set(first.(priors))
        extra_keys = setdiff(user_symbols, default_symbols)

        isempty(extra_keys) || throw(ArgumentError("Unknown prior symbols: $(extra_keys)"))
    end

    merged_dict = Dict{Symbol,Distribution}(default_priors)
    for (sym, dist) in priors
        merged_dict[sym] = dist
    end

    merged_priors = sort(collect(Pair.(keys(merged_dict), values(merged_dict))), by = first)

    return distprod(; merged_priors...)
end

"""
    build_posterior(data::SpectrumData, priors::NamedTupleDist, configs::FitConfigs)

Construct a posterior measure from observed data and a prior distribution.

# Arguments
- `data::SpectrumData`: the observed spectrum data
- `priors`: the prior distribution (result of [`build_prior`](@ref))
- `configs::FitConfigs`: Fitting configurations

# Returns
- A `PosteriorMeasure` wrapping the log-likelihood and prior

# See also
- [`build_prior`](@ref) for constructing the prior
- [`poisson_ll`](@ref) for the likelihood function
- [`ModelParams`](@ref) for the model parameter structure
"""
function build_posterior(data::SpectrumData, priors::NamedTupleDist, configs::FitConfigs)

    function _log_likelihood(params::NamedTuple)
        model_params = ModelParams(params)
        return poisson_ll(data, model_params, configs)
    end

    return PosteriorMeasure(logfuncdensity(_log_likelihood), priors)
end
