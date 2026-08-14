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
    AbstractIntegrationMethod

Abstract supertype for the integration methods used to compute the expected bin counts in 
[`poisson_ll`](@ref).

Concrete subtypes are dispatched on by [`_integrate`](@ref) to select how the model is 
integrated over each energy bin.

# See also
- [`Analytical`](@ref), [`Numerical`](@ref), [`Midpoint`](@ref)
"""
abstract type AbstractIntegrationMethod end

"""
    Analytical

Integrate the model analytically over each energy bin.

# See also
- [`AbstractIntegrationMethod`](@ref), [`Numerical`](@ref), [`Midpoint`](@ref)
"""
struct Analytical <: AbstractIntegrationMethod end

"""
    Numerical

Integrate the model numerically over each energy bin with `QuadGK.quadgk`.

# See also
- [`AbstractIntegrationMethod`](@ref), [`Analytical`](@ref), [`Midpoint`](@ref)
"""
struct Numerical <: AbstractIntegrationMethod end

"""
    Midpoint

Integrate the model via the midpoint rule over each energy bin.

# See also
- [`AbstractIntegrationMethod`](@ref), [`Analytical`](@ref), [`Numerical`](@ref)
"""
struct Midpoint <: AbstractIntegrationMethod end

"""
    FitConfigs{T<:AbstractFloat}

Configuration options for the fitting process.

`mu` and `sigma` are required and have no default values. All remaining fields default to 
reasonable values.

# Fields
- `mu::T`: expected centroid position of the peak in keV
- `sigma::T`: expected standard deviation of the Gaussian core in keV
- `integration_method::AbstractIntegrationMethod`: the integration method used to compute 
  expected bin counts. Either `Analytical()`, `Numerical()`, or `Midpoint()`. 
  Default: `Analytical()`

# See also
- [`PriorConfigs`](@ref) for the configuration options for the priors
- [`AbstractIntegrationMethod`](@ref) for the available integration methods
"""
Base.@kwdef struct FitConfigs{T<:AbstractFloat}
    mu::T
    sigma::T
    integration_method::AbstractIntegrationMethod = Analytical()
    prior::PriorConfigs = PriorConfigs()
end
