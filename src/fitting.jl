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
- `-Inf` if any expected counts are negative (unphysical model configuration)
- sum of log-likelihoods across bins (total log-likelihood) otherwise

# See also
- [`full_model`](@ref) for the underlying model
- [`ModelParams`](@ref) for the parameter structure
"""
function poisson_ll(data::SpectrumData, params::ModelParams)
    expected_counts = first.(
        quadgk.(
            x -> full_model(x, params),
            data.bin_centers .- data.bin_size/2,
            data.bin_centers .+ data.bin_size/2,
        ),
    )
    if any(expected_counts .< 0)
        return -Inf
    end
    result_vector = logpdf.(Poisson.(expected_counts), data.weights)
    return sum(result_vector)
end

"""
    build_prior(
        params::ModelParams,
        mu::Real,
        sigma::Real;
        peak_height::Union{<:Real,Nothing},
        peak_area::Union{<:Real,Nothing},
    )

Construct a prior distribution over the model parameters for Bayesian fitting.

Components that are `false` (or `nothing` for container fields) are skipped. Components set 
to `true` or to a concrete parameter struct receive a weakly informative prior.

# Arguments
- `params::ModelParams`: model specification indicating which components are enabled
- `mu::Real`: expected centroid position of the peak in keV
- `sigma::Real`: expected standard deviation of the Gaussian core in keV
- `peak_height::Union{<:Real,Nothing}`: approximate peak height in counts/keV
- `peak_area::Union{<:Real,Nothing}`: approximate integrated peak area in counts

# Returns
- A `NamedTupleDist` (via `distprod`) over the enabled component parameters

# Throws
- An `ArgumentError` if both `params.peak` and `params.background` are `nothing`
- An `ArgumentError` if either `peak_height` or `peak_area` where not supplied when they 
  were needed

# Details

The following priors are defined per enabled component:

| Symbol | Prior | Component |
| --- | --- | --- |
| `:mu` | `Normal(mu, 0.6)` | all |
| `:sigma` | `truncated(Normal(sigma, 0.6), 0, Inf)` | `peak.gaussian`, `peak.lowEnergyTail`, `peak.highEnergyTail` |
| `:gaussian_A` | `Uniform(0, peak_area)` | `peak.gaussian` |
| `:compton_h` | `Uniform(0, peak_height)` | `peak.compton` |
| `:lowEnergyTail_A` | `Uniform(0, peak_area)` | `peak.lowEnergyTail` |
| `:lowEnergyTail_tau` | Uniform(eps(), 10) | `peak.lowEnergyTail` |
| `:highEnergyTail_A` | `Uniform(0, peak_area)` | `peak.highEnergyTail` |
| `:highEnergyTail_tau` | Uniform(eps(), 10) | `peak.highEnergyTail` |
| `:quadPoly_C` | `Uniform(-1, 1)` | `background.quadPoly` |
| `:linPoly_C` | `Uniform(-10, 10)` | `background.linPoly` |
| `:constPoly_C` | `Uniform(0, peak_height)` | `background.constPoly` |

# See also
- [`ModelParams`](@ref), [`PeakParams`](@ref), [`BackgroundParams`](@ref) for the model
  specification
- [`poisson_ll`](@ref) for the likelihood that uses these priors
"""
function build_prior(
    params::ModelParams,
    mu::Real,
    sigma::Real;
    peak_height::Union{<:Real,Nothing} = nothing,
    peak_area::Union{<:Real,Nothing} = nothing,
)
    peak_params = params.peak
    background_params = params.background

    if (isnothing(peak_params) && isnothing(background_params))
        throw(ArgumentError("No model specified"))
    end

    priors = []

    push!(priors, :mu => Normal(mu, 0.6))

    has_gaussian_or_tail =
        !isnothing(peak_params) && (
            peak_params.gaussian !== false ||
            peak_params.lowEnergyTail !== false ||
            peak_params.highEnergyTail !== false
        )
    if has_gaussian_or_tail
        push!(priors, :sigma => truncated(Normal(sigma, 0.6), 0, Inf))
    end

    if !isnothing(peak_params)
        if peak_params.gaussian !== false
            isnothing(peak_area) &&
                throw(ArgumentError("`peak_area` required for `gaussian` component"))
            push!(priors, :gaussian_A => Uniform(0, peak_area))
        end

        if peak_params.compton !== false
            isnothing(peak_height) &&
                throw(ArgumentError("`peak_height` required for `compton` component"))
            push!(priors, :compton_h => Uniform(0, peak_height))
        end

        if peak_params.lowEnergyTail !== false
            isnothing(peak_area) &&
                throw(ArgumentError("`peak_area` required for `lowEnergyTail` component"))
            push!(priors, :lowEnergyTail_A => Uniform(0, peak_area))
            push!(priors, :lowEnergyTail_tau => Uniform(eps(), 10))
        end

        if peak_params.highEnergyTail !== false
            isnothing(peak_area) &&
                throw(ArgumentError("`peak_area` required for `highEnergyTail` component"))
            push!(priors, :highEnergyTail_A => Uniform(0, peak_area))
            push!(priors, :highEnergyTail_tau => Uniform(eps(), 10))
        end
    end

    if !isnothing(background_params)
        background_params.quadPoly !== false && push!(priors, :quadPoly_C => Uniform(-1, 1))
        background_params.linPoly !== false && push!(priors, :linPoly_C => Uniform(-10, 10))

        if background_params.constPoly !== false
            isnothing(peak_height) &&
                throw(ArgumentError("`peak_height` required for `constPoly` component"))
            push!(priors, :constPoly_C => Uniform(0, peak_height))
        end
    end

    return distprod(; priors...)
end

"""
    build_posterior(data::SpectrumData, priors::NamedTupleDist)

Construct a posterior measure from observed data and a prior distribution.

`hasproperty` checks are evaluated once at construction time; the log-likelihood closure
only checks pre-computed `Bool` flags on each sample.

# Arguments
- `data::SpectrumData`: the observed spectrum data
- `priors`: the prior distribution (result of [`build_prior`](@ref))

# Returns
- A `PosteriorMeasure` wrapping the log-likelihood and prior, ready for
  [`bat_sample`](@ref).

# See also
- [`build_prior`](@ref) for constructing the prior
- [`poisson_ll`](@ref) for the likelihood function
- [`ModelParams`](@ref) for the model parameter structure
"""
function build_posterior(data::SpectrumData, priors::NamedTupleDist)
    has_gaussian = hasproperty(priors, :gaussian_A)
    has_compton = hasproperty(priors, :compton_h)
    has_lowEnergyTail = hasproperty(priors, :lowEnergyTail_tau)
    has_highEnergyTail = hasproperty(priors, :highEnergyTail_tau)
    has_quadPoly = hasproperty(priors, :quadPoly_C)
    has_linPoly = hasproperty(priors, :linPoly_C)
    has_constPoly = hasproperty(priors, :constPoly_C)

    has_peak = has_gaussian || has_compton || has_lowEnergyTail || has_highEnergyTail
    has_background = has_quadPoly || has_linPoly || has_constPoly

    # Log-likelihood closure called by the BAT sampler with a NamedTuple of parameter
    # values. Uses pre-computed Bool flags to determine which components to assemble.
    function _log_likelihood(params::NamedTuple)
        if has_peak
            gaussian_params =
                has_gaussian ?
                GaussianParams(
                    A = params.gaussian_A,
                    mu = params.mu,
                    sigma = params.sigma,
                ) : false

            compton_params =
                has_compton ?
                ComptonParams(h = params.compton_h, mu = params.mu, sigma = params.sigma) :
                false

            lowEnergyTail_params =
                has_lowEnergyTail ?
                ExGaussianParams(
                    A = params.lowEnergyTail_A,
                    tau = params.lowEnergyTail_tau,
                    is_lowEnergyTail = true,
                    mu = params.mu,
                    sigma = params.sigma,
                ) : false
            highEnergyTail_params =
                has_highEnergyTail ?
                ExGaussianParams(
                    A = params.heighEnergyTail_A,
                    tau = params.highEnergyTail_tau,
                    is_lowEnergyTail = false,
                    mu = params.mu,
                    sigma = params.sigma,
                ) : false

            peak = PeakParams(
                gaussian = gaussian_params,
                compton = compton_params,
                lowEnergyTail = lowEnergyTail_params,
                highEnergyTail = highEnergyTail_params,
            )
        else
            peak = nothing
        end

        if has_background
            quadPoly_params =
                has_quadPoly ? QuadPolyParams(C = params.quadPoly_C, mu = params.mu) : false

            linPoly_params =
                has_linPoly ? LinPolyParams(C = params.linPoly_C, mu = params.mu) : false

            constPoly_params =
                has_constPoly ? ConstPolyParams(C = params.constPoly_C) : false

            background = BackgroundParams(
                quadPoly = quadPoly_params,
                linPoly = linPoly_params,
                constPoly = constPoly_params,
            )
        else
            background = nothing
        end

        model_params = ModelParams(peak = peak, background = background)
        return LogDVal(poisson_ll(data, model_params))
    end

    return PosteriorMeasure(_log_likelihood, priors)
end
