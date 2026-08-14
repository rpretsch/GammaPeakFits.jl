"""
    PriorConfigs{T<:AbstractFloat}

Configuration options for the priors.

# Fields
- `mu_std::T`: standard deviation of the prior on `:mu` in keV. Default: `0.6`
- `sigma_std::T`: standard deviation of the prior on `:sigma` in keV. Default: `0.6`
- `lowEnergyTail_tau_upper::T`: upper bound of the `Uniform` prior on `:lowEnergyTail_tau` 
  in keV. Default: `10.0`
- `highEnergyTail_tau_upper::T`: upper bound of the `Uniform` prior on 
  `:highEnergyTail_tau` in keV. Default: `10.0`
- `quadPoly_C_limits::Tuple{T,T}`: `(lower, upper)` bounds of the `Uniform` prior on 
  `:quadPoly_C` in counts/keV³. Default: `(-1.0, 1.0)`
- `linPoly_C_limits::Tuple{T,T}`: `(lower, upper)` bounds of the `Uniform` prior on 
  `:linPoly_C` in counts/keV². Default: `(-10.0, 10.0)`

# See also
- [`build_prior`](@ref) which consumes these configurations
"""
Base.@kwdef struct PriorConfigs{T<:AbstractFloat}
    mu_std::T = 0.6
    sigma_std::T = 0.6
    lowEnergyTail_tau_upper::T = 10.0
    highEnergyTail_tau_upper::T = 10.0
    quadPoly_C_limits::Tuple{T,T} = (-1.0, 1.0)
    linPoly_C_limits::Tuple{T,T} = (-10.0, 10.0)
end

"""
    FitConfigs{T<:AbstractFloat}

Configuration options for the fitting process.

`mu` and `sigma` are required and have no default values. All remaining fields default to 
reasonable values.

# Fields
- `mu::T`: expected centroid position of the peak in keV
- `sigma::T`: expected standard deviation of the Gaussian core in keV
- `integration_method::Symbol`: the integration method used to compute expected bin counts.
  Either `:analytical`, `:numerical`, or `:midpoint`. Default: `:analytical`

# See also
- [`PriorConfigs`](@ref) for the configuration options for the priors
"""
Base.@kwdef struct FitConfigs{T<:AbstractFloat}
    mu::T
    sigma::T
    integration_method::Symbol = :analytical
    prior::PriorConfigs = PriorConfigs()
end
