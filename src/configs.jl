"""
    PriorPair

Shorthand for `Pair{Symbol,<:Distribution}`, used for the custom priors in
[`FitConfigs`](@ref).

# See also
- [`FitConfigs`](@ref)
- [`build_prior`](@ref)
"""
const PriorPair = Pair{Symbol,<:Distribution}

"""
    FitConfigs

Configuration options for the fitting process.

`mu` and `sigma` are required and have no default values. All remaining fields default to
reasonable values. The configuration acts as the single source of truth for the fit window,
the prior centers, and the prior overrides.

# Fields
- `mu::Float64`: expected centroid position of the peak in keV
- `sigma::Float64`: expected standard deviation of the Gaussian core in keV
- `window_size::Float64`: full width of the fit window in keV, centered on `mu`. Should be
  large enough to contain the `mu +- 3 * sigma` peak region. Default: `10 * sigma`
- `priors::Vector{PriorPair}`: custom priors that overwrite the defaults built by
  [`build_prior`](@ref) for the specified symbols. Default: `PriorPair[]`

# See also
- [`build_prior`](@ref) for the prior construction that consumes these options
"""
Base.@kwdef struct FitConfigs
    mu::Float64
    sigma::Float64
    window_size::Float64 = 10 * sigma
    priors::Vector{PriorPair} = PriorPair[]
end
