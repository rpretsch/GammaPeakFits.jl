"""
    AbstractIntegrationMethod

Abstract supertype for the integration methods used to compute the expected bin counts in 
[`poisson_ll`](@ref).

Concrete subtypes are dispatched on by [`_expected_counts`](@ref) to select how the model 
is integrated over each energy bin.

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
    FitConfigs

Configuration options for the fitting process.

`mu` and `sigma` are required and have no default values. All remaining fields default to 
reasonable values.

# Fields
- `mu::Float64`: expected centroid position of the peak in keV
- `sigma::Float64`: expected standard deviation of the Gaussian core in keV
- `integration_method::AbstractIntegrationMethod`: the integration method used to compute 
  expected bin counts. Either `Analytical()`, `Numerical()`, or `Midpoint()`. 
  Default: `Analytical()`

# See also
- [`AbstractIntegrationMethod`](@ref) for the available integration methods
"""
Base.@kwdef struct FitConfigs
    mu::Float64
    sigma::Float64
    integration_method::AbstractIntegrationMethod = Analytical()
end
