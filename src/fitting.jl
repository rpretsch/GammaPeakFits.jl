"""
    poisson_ll(
        bin_center::Union{Real, AbstractVector{<:Real}}, 
        weight::Union{Int, AbstractVector{<:Int}}, 
        params::ModelParams, 
        bin_size::Real,
    )

Compute the Poisson log-likelihood for the model given observed counts.

The expected count in each bin is obtained by integrating [`full_model`](@ref) over the bin 
width `bin_size` using `QuadGK.jl`.
The log-probability of the observed integer count `weight` is then evaluated under a
Poisson distribution with that expected rate.

# Arguments
- `bin_center::Union{Real, AbstractVector{<:Real}}`: bin center(s) in keV
- `weight::Union{Int, AbstractVector{<:Int}}`: observed count(s) per bin
- `params::ModelParams`: model parameters
- `bin_size::Real`: width of each bin (used for integration bounds) in keV

# Returns
- Scalar: the log-likelihood for a single bin
- Vector: sum of log-likelihoods across bins (total log-likelihood)

# See also
- [`full_model`](@ref) for the underlying model
- [`ModelParams`](@ref) for the parameter structure
"""
function poisson_ll(bin_center::Real, weight::Int, params::ModelParams, bin_size::Real)
    expected_count = quadgk(
        x -> full_model(x, params),
        bin_center - bin_size/2,
        bin_center + bin_size/2,
    )[1]
    return logpdf.(Poisson(expected_count), weight)
end
function poisson_ll(
    bin_center::AbstractVector{<:Real},
    weight::AbstractVector{<:Int},
    params::ModelParams,
    bin_size::Real,
)
    result_vector = poisson_ll.(bin_center, weight, Ref(params), bin_size)
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
- A `ProductDistribution` (via `distprod`) over the enabled component parameters

# Throws
- `ArgumentError` if both `params.peak` and `params.background` are `nothing`

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
p = PeakParams{Float64}(gaussian = true, compton = true)
b = BackgroundParams{Float64}(b0 = true)
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
        params.background.b2 !== false && (priors[:background_b2] = Uniform(-1, 1))
        params.background.b1 !== false && (priors[:background_b1] = Uniform(-10, 10))
        params.background.b0 !== false && (priors[:background_b0] = Uniform(0, peak_height))
    end

    return distprod(; priors...)
end
