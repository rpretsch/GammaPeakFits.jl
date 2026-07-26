"""
    gaussian(x::Union{Real, AbstractVector{<:Real}}, params::GaussianParams)

Evaluate a scaled Gaussian (normal) distribution at `x`.

Uses `Distributions.jl` for the normal distribution implementation.

# Arguments
- `x::Union{Real, AbstractVector{<:Real}}`: position(s) at which to evaluate in keV
- `params::GaussianParams`: component parameters

# Returns
- Scalar: the evaluated Gaussian amplitude at `x` in counts/keV
- Vector: an array of evaluated Gaussian amplitudes at each element of `x` counts/keV

# See also
[`GaussianParams`](@ref) for the parameters.
"""
function gaussian(x::Real, params::GaussianParams)
    return params.A * pdf(Normal(params.mu, params.sigma), x)
end
function gaussian(x::AbstractVector{<:Real}, params::GaussianParams)
    return gaussian.(x, Ref(params))
end

"""
    compton(x::Union{Real, AbstractVector{<:Real}}, params::ComptonParams)

Evaluate a Compton-edge step function component at `x`.

Uses the complementary error function (`erfc`) from `SpecialFunctions.jl` to model the
smooth step from Compton scattering.

# Arguments
- `x::Union{Real, AbstractVector{<:Real}}`: position(s) at which to evaluate in keV
- `params::ComptonParams`: component parameters

# Returns
- Scalar: the evaluated Compton step height at `x` in counts/keV
- Vector: an array of evaluated Compton step heights at each element of `x` in counts/keV

# See also
[`ComptonParams`](@ref) for the parameters.
"""
function compton(x::Real, params::ComptonParams)
    return params.h/2 * erfc((x - params.mu)/(sqrt(2) * params.sigma))
end
function compton(x::AbstractVector{<:Real}, params::ComptonParams)
    return compton.(x, Ref(params))
end

"""
    exGaussian(x::Union{Real, AbstractVector{<:Real}}, params::ExGaussianParams)

Evaluate an exponentially modified Gaussian (ex-Gaussian) tail component at `x`.

Used to model asymmetric peak tailing.
Uses the complementary error function (`erfc`) from `SpecialFunctions.jl`.

# Arguments
- `x::Union{Real, AbstractVector{<:Real}}`: position(s) at which to evaluate in keV
- `params::ExGaussianParams`: component parameters

# Returns
- Scalar: the evaluated ex-Gaussian tail amplitude at `x` in counts/keV
- Vector: an array of evaluated tail amplitudes at each element of `x` in counts/keV

# See also
[`ExGaussianParams`](@ref) for the parameters.
"""
function exGaussian(x::Real, params::ExGaussianParams)
    return params.A * params.lambda/2 *
           exp(params.lambda/2 * (2 * params.mu + params.lambda * params.sigma^2 - 2 * x)) *
           erfc((params.mu + params.lambda * params.sigma^2 - x)/(sqrt(2) * params.sigma))
end
function exGaussian(x::AbstractVector{<:Real}, params::ExGaussianParams)
    return exGaussian.(x, Ref(params))
end

"""
    background_model(x::Union{Real, AbstractVector{<:Real}}, params::BackgroundParams)

Evaluate a quadratic polynomial background at `x`.

The polynomial is centred at `params.mu` to improve the numerical stability during fitting.
Each term is optional — set the corresponding field to `nothing` in 
[`BackgroundParams`](@ref) to exclude it.

# Arguments
- `x::Union{Real, AbstractVector{<:Real}}`: position(s) at which to evaluate in keV
- `params::BackgroundParams`: background parameters

# Returns
- Scalar: the evaluated quadratic background value at `x` in counts/keV
- Vector: an array of evaluated background values at each element of `x` in counts/keV

# See also
[`BackgroundParams`](@ref) for the parameters.
"""
function background_model(x::Real, params::BackgroundParams)
    result = zero(float(x))
    isnothing(params.b2) || (result += params.b2 * (x - params.mu)^2)
    isnothing(params.b1) || (result += params.b1 * (x - params.mu))
    isnothing(params.b0) || (result += params.b0)
    return result
end
function background_model(x::AbstractVector{<:Real}, params::BackgroundParams)
    result = zeros(float(eltype(x)), length(x))
    isnothing(params.b2) || (result .+= params.b2 * (x - params.mu)^2)
    isnothing(params.b1) || (result .+= params.b1 * (x - params.mu))
    isnothing(params.b0) || (result .+= params.b0)
    return result
end

"""
    peak_model(x::Union{Real, AbstractVector{<:Real}}, params::PeakParams)

Evaluate the combined peak shape (Gaussian + Compton edge + ex-Gaussian tails) at `x`.

Each term is optional — set the corresponding field to `nothing` in 
[`PeakParams`](@ref) to exclude it.

# Arguments
- `x::Union{Real, AbstractVector{<:Real}}`: position(s) at which to evaluate in keV
- `params::PeakParams`: peak component parameters

# Returns
- Scalar: the summed peak amplitude at `x` in counts/keV
- Vector: an array of summed peak amplitudes at each element of `x` in counts/keV

# See also
- [`PeakParams`](@ref) for the parameters
- [`gaussian`](@ref), [`compton`](@ref), and [`exGaussian`](@ref) for the components
"""
function peak_model(x::Real, params::PeakParams)
    result = zero(float(x))
    isnothing(params.gaussian) || (result += gaussian(x, params.gaussian))
    isnothing(params.compton) || (result += compton(x, params.compton))
    isnothing(params.highEnergyTail) || (result += exGaussian(x, params.highEnergyTail))
    isnothing(params.lowEnergyTail) || (result += exGaussian(x, params.lowEnergyTail))
    return result
end
function peak_model(x::AbstractVector{<:Real}, params::PeakParams)
    result = zeros(float(eltype(x)), length(x))
    isnothing(params.gaussian) || (result .+= gaussian(x, params.gaussian))
    isnothing(params.compton) || (result .+= compton(x, params.compton))
    isnothing(params.highEnergyTail) || (result .+= exGaussian(x, params.highEnergyTail))
    isnothing(params.lowEnergyTail) || (result .+= exGaussian(x, params.lowEnergyTail))
    return result
end

"""
    full_model(x::Union{Real, AbstractVector{<:Real}}, params::ModelParams)

Evaluate the complete gamma-peak model (peak shape + background) at `x`.

Combines [`peak_model`](@ref) and [`background_model`](@ref). Each term is optional — set
the corresponding field to `nothing` in [`ModelParams`](@ref) to exclude it.

# Arguments
- `x::Union{Real, AbstractVector{<:Real}}`: position(s) at which to evaluate in keV
- `params::ModelParams`: full model parameters

# Returns
- Scalar: the evaluated model value (peak + background) at `x` in counts/keV
- Vector: an array of total model values at each element of `x` in counts/keV

# Examples

```julia
mu = 2048.0
sigma = 10.0
p = PeakParams(gaussian = GaussianParams(A = 100.0, mu = mu, sigma = sigma))
b = BackgroundParams(b0 = 100.0, mu = mu)
m = ModelParams(peak = p, background = b)
full_model(1:4096, m)
```

# See also
- [`ModelParams`](@ref) for the parameters
- [`peak_model`](@ref), and [`background_model`](@ref) for the components
"""
function full_model(x::Real, params::ModelParams)
    result = zero(float(x))
    isnothing(params.peak) || (result += peak_model(x, params.peak))
    isnothing(params.background) || (result += background_model(x, params.background))
    return result
end
function full_model(x::AbstractVector{<:Real}, params::ModelParams)
    result = zeros(float(eltype(x)), length(x))
    isnothing(params.peak) || (result .+= peak_model(x, params.peak))
    isnothing(params.background) || (result .+= background_model(x, params.background))
    return result
end
