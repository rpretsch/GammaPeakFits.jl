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
        params::ModelParams,
        configs::FitConfigs;
        peak_height::Union{<:Real,Nothing},
        peak_area::Union{<:Real,Nothing},
    )

Construct a prior distribution over the model parameters for Bayesian fitting.

Components that are `Disabled()` are skipped. Components set to `Enabled()` or to a 
concrete parameter struct receive a weakly informative prior.

The expected peak centroid `configs.mu` and core width `configs.sigma`, as well as all prior
widths and bounds, are taken from [`FitConfigs`](@ref) and can be tuned there.

# Arguments
- `params::ModelParams`: model specification indicating which components are enabled
- `configs::FitConfigs`: fitting configuration
- `peak_height::Union{<:Real,Nothing}`: approximate peak height in counts/keV. Optional for 
  some model configurations. Default: `nothing`
- `peak_area::Union{<:Real,Nothing}`: approximate integrated peak area in counts. Optional 
  for some model configurations. Default: `nothing`

# Returns
- A `NamedTupleDist` (via `distprod`) over the enabled component parameters

# Throws
- An `ArgumentError` if both `params.peak` and `params.background` are `Disabled()`
- An `ArgumentError` if either `peak_height` or `peak_area` were not supplied when they 
  were needed

# Details

The following priors are defined per enabled component:

| Symbol | Prior | Component |
| --- | --- | --- |
| `:mu` | `Normal(configs.mu, prior_configs.mu_std)` | all except `background.constPoly` |
| `:sigma` | `truncated(Normal(configs.sigma, prior_configs.sigma_std), eps(), Inf)` | all `peak` components |
| `:gaussian_A` | `Uniform(0, peak_area)` | `peak.gaussian` |
| `:compton_h` | `Uniform(0, peak_height)` | `peak.compton` |
| `:lowEnergyTail_A` | `Uniform(0, peak_area)` | `peak.lowEnergyTail` |
| `:lowEnergyTail_tau` | `Uniform(eps(), prior_configs.lowEnergyTail_tau_upper)` | `peak.lowEnergyTail` |
| `:highEnergyTail_A` | `Uniform(0, peak_area)` | `peak.highEnergyTail` |
| `:highEnergyTail_tau` | `Uniform(eps(), prior_configs.highEnergyTail_tau_upper)` | `peak.highEnergyTail` |
| `:quadPoly_C` | `Uniform(prior_configs.quadPoly_C_limits)` | `background.quadPoly` |
| `:linPoly_C` | `Uniform(prior_configs.linPoly_C_limits)` | `background.linPoly` |
| `:constPoly_C` | `Uniform(0, peak_height)` | `background.constPoly` |

# See also
- [`Enabled`](@ref), [`Disabled`](@ref), [`AbstractComponent`](@ref) for the component 
  management
- [`FitConfigs`](@ref) for tuning the prior centers, widths, and bounds
- [`ModelParams`](@ref), [`PeakParams`](@ref), [`BackgroundParams`](@ref) for the model
  specification
- [`poisson_ll`](@ref) for the likelihood that uses these priors
"""
function build_prior(
    params::ModelParams,
    configs::FitConfigs;
    peak_height::Union{<:Real,Nothing} = nothing,
    peak_area::Union{<:Real,Nothing} = nothing,
)
    peak_params = params.peak
    background_params = params.background
    prior_configs = configs.prior

    if (!is_present(peak_params) && !is_present(background_params))
        throw(ArgumentError("No model specified"))
    end

    priors = []
    needs_mu =
        is_present(peak_params) && (
            is_present(peak_params.gaussian) ||
            is_present(peak_params.lowEnergyTail) ||
            is_present(peak_params.highEnergyTail) ||
            is_present(peak_params.compton)
        ) ||
        is_present(background_params) &&
        (is_present(background_params.quadPoly) || is_present(background_params.linPoly))
    needs_mu && push!(priors, :mu => Normal(configs.mu, prior_configs.mu_std))

    needs_sigma =
        is_present(peak_params) && (
            is_present(peak_params.gaussian) ||
            is_present(peak_params.lowEnergyTail) ||
            is_present(peak_params.highEnergyTail) ||
            is_present(peak_params.compton)
        )
    needs_sigma && push!(
        priors,
        :sigma => truncated(Normal(configs.sigma, prior_configs.sigma_std), eps(), Inf),
    )

    if is_present(peak_params)
        if is_present(peak_params.gaussian)
            isnothing(peak_area) &&
                throw(ArgumentError("`peak_area` required for `gaussian` component"))
            push!(priors, :gaussian_A => Uniform(0, peak_area))
        end

        if is_present(peak_params.compton)
            isnothing(peak_height) &&
                throw(ArgumentError("`peak_height` required for `compton` component"))
            push!(priors, :compton_h => Uniform(0, peak_height))
        end

        if is_present(peak_params.lowEnergyTail)
            isnothing(peak_area) &&
                throw(ArgumentError("`peak_area` required for `lowEnergyTail` component"))
            push!(priors, :lowEnergyTail_A => Uniform(0, peak_area))
            push!(
                priors,
                :lowEnergyTail_tau => Uniform(eps(), prior_configs.lowEnergyTail_tau_upper),
            )
        end

        if is_present(peak_params.highEnergyTail)
            isnothing(peak_area) &&
                throw(ArgumentError("`peak_area` required for `highEnergyTail` component"))
            push!(priors, :highEnergyTail_A => Uniform(0, peak_area))
            push!(
                priors,
                :highEnergyTail_tau =>
                    Uniform(eps(), prior_configs.highEnergyTail_tau_upper),
            )
        end
    end

    if is_present(background_params)
        is_present(background_params.quadPoly) &&
            push!(priors, :quadPoly_C => Uniform(prior_configs.quadPoly_C_limits...))
        is_present(background_params.linPoly) &&
            push!(priors, :linPoly_C => Uniform(prior_configs.linPoly_C_limits...))

        if is_present(background_params.constPoly)
            isnothing(peak_height) &&
                throw(ArgumentError("`peak_height` required for `constPoly` component"))
            push!(priors, :constPoly_C => Uniform(0, peak_height))
        end
    end

    return distprod(; priors...)
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
