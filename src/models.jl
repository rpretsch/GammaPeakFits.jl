"""
    gaussian(x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}, params::GaussianParams)

Evaluate a scaled Gaussian (normal) distribution at `x`.

Uses `Distributions.jl` for the normal distribution implementation.

# Arguments
- `x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}`: position(s) at which to 
  evaluate in keV
- `params::GaussianParams`: component parameters

# Returns
- Scalar: the evaluated Gaussian amplitude at `x` in counts/keV
- Vector: an array of evaluated Gaussian amplitudes at each element of `x` counts/keV

# See also
[`GaussianParams`](@ref) for the parameters.
"""
function gaussian(x::AbstractFloat, params::GaussianParams)
    return params.A * pdf(Normal(params.mu, params.sigma), x)
end
function gaussian(x::AbstractVector{<:AbstractFloat}, params::GaussianParams)
    return gaussian.(x, Ref(params))
end

"""
    compton(x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}, params::ComptonParams)

Evaluate a Compton-edge step function component at `x`.

Uses the complementary error function (`erfc`) from `SpecialFunctions.jl` to model the
smooth step from Compton scattering.

# Arguments
- `x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}`: position(s) at which to 
  evaluate in keV
- `params::ComptonParams`: component parameters

# Returns
- Scalar: the evaluated Compton step height at `x` in counts/keV
- Vector: an array of evaluated Compton step heights at each element of `x` in counts/keV

# See also
[`ComptonParams`](@ref) for the parameters.
"""
function compton(x::AbstractFloat, params::ComptonParams)
    return params.h/2 * erfc((x - params.mu)/(sqrt(2) * params.sigma))
end
function compton(x::AbstractVector{<:AbstractFloat}, params::ComptonParams)
    return compton.(x, Ref(params))
end

"""
    exGaussian(
        x::Union{AbstractFloat, 
        AbstractVector{<:AbstractFloat}}, 
        params::ExGaussianParams,
    )

Evaluate an exponentially modified Gaussian (ex-Gaussian) tail component at `x`.

Used to model asymmetric peak tailing.
Uses the complementary error function (`erfc`) from `SpecialFunctions.jl`.

# Arguments
- `x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}`: position(s) at which to 
  evaluate in keV
- `params::ExGaussianParams`: component parameters

# Returns
- Scalar: the evaluated ex-Gaussian tail amplitude at `x` in counts/keV
- Vector: an array of evaluated tail amplitudes at each element of `x` in counts/keV

# See also
[`ExGaussianParams`](@ref) for the parameters.
"""
function exGaussian(x::AbstractFloat, params::ExGaussianParams)
    return params.A * params.lambda/2 *
           exp(params.lambda/2 * (2 * params.mu + params.lambda * params.sigma^2 - 2 * x)) *
           erfc((params.mu + params.lambda * params.sigma^2 - x)/(sqrt(2) * params.sigma))
end
function exGaussian(x::AbstractVector{<:AbstractFloat}, params::ExGaussianParams)
    return exGaussian.(x, Ref(params))
end

"""
    peak_model(x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}, params::PeakParams)

Evaluate the combined peak shape (Gaussian + Compton edge + ex-Gaussian tails) at `x`.

Each term is optional — set the corresponding field to `false` in [`PeakParams`](@ref) to 
exclude it.

# Arguments
- `x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}`: position(s) at which to 
  evaluate in keV
- `params::PeakParams`: peak component parameters

# Returns
- Scalar: the summed peak amplitude at `x` in counts/keV
- Vector: an array of summed peak amplitudes at each element of `x` in counts/keV

# Examples

```julia
mu = 2048.0
sigma = 10.0
p = PeakParams(gaussian = GaussianParams(A = 100.0, mu = mu, sigma = sigma))
peak_model(1.0:4096.0, p)
```

# See also
- [`PeakParams`](@ref) for the parameters
- [`gaussian`](@ref), [`compton`](@ref), and [`exGaussian`](@ref) for the components
"""
function peak_model(x::AbstractFloat, params::PeakParams)
    result = zero(float(x))
    isa(params.gaussian, GaussianParams) && (result += gaussian(x, params.gaussian))
    isa(params.compton, ComptonParams) && (result += compton(x, params.compton))
    isa(params.highEnergyTail, ExGaussianParams) &&
        (result += exGaussian(x, params.highEnergyTail))
    isa(params.lowEnergyTail, ExGaussianParams) &&
        (result += exGaussian(x, params.lowEnergyTail))
    return result
end
function peak_model(x::AbstractVector{<:AbstractFloat}, params::PeakParams)
    result = zeros(float(eltype(x)), length(x))
    isa(params.gaussian, GaussianParams) && (result .+= gaussian(x, params.gaussian))
    isa(params.compton, ComptonParams) && (result .+= compton(x, params.compton))
    isa(params.highEnergyTail, ExGaussianParams) &&
        (result .+= exGaussian(x, params.highEnergyTail))
    isa(params.lowEnergyTail, ExGaussianParams) &&
        (result .+= exGaussian(x, params.lowEnergyTail))
    return result
end

"""
    quad_polynomial(
        x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}},
        params::QuadPolyParams,
    )

Evaluate a scaled quadratic polynomial term at `x`.

# Arguments
- `x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}`: position(s) at which to
  evaluate in keV
- `params::QuadPolyParams`: component parameters

# Returns
- Scalar: the evaluated quadratic polynomial amplitude at `x` in counts/keV
- Vector: an array of evaluated quadratic polynomial amplitudes at each element of `x` in 
  counts/keV

# See also
[`QuadPolyParams`](@ref) for the parameter structure.
"""
function quad_polynomial(x::AbstractFloat, params::QuadPolyParams)
    return params.C * (x - params.mu)^2
end
function quad_polynomial(x::AbstractVector{<:AbstractFloat}, params::QuadPolyParams)
    return quad_polynomial.(x, Ref(params))
end

"""
    lin_polynomial(
        x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}},
        params::LinPolyParams,
    )

Evaluate a scaled linear polynomial term at `x`.

# Arguments
- `x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}`: position(s) at which to
  evaluate in keV
- `params::LinPolyParams`: component parameters

# Returns
- Scalar: the evaluated linear polynomial amplitude at `x` in counts/keV
- Vector: an array of evaluated linear polynomial amplitudes at each element of `x` in 
  counts/keV

# See also
[`LinPolyParams`](@ref) for the parameter structure.
"""
function lin_polynomial(x::AbstractFloat, params::LinPolyParams)
    return params.C * (x - params.mu)
end
function lin_polynomial(x::AbstractVector{<:AbstractFloat}, params::LinPolyParams)
    return lin_polynomial.(x, Ref(params))
end

"""
    const_polynomial(
        x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}},
        params::ConstPolyParams
    )

Evaluate a constant polynomial term.

Uses `x` to set wether to return as a scalar or vector.

# Arguments
- `x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}`: sets the return type
- `params::ConstPolyParams`: polynomial parameters

# Returns
- Scalar: the constant coefficient in counts/keV
- Vector: an array of constant coefficient in counts/keV

# See also
[`ConstPolyParams`](@ref) for the parameter structure.
"""
function const_polynomial(x::AbstractFloat, params::ConstPolyParams)
    return params.C
end
function const_polynomial(x::AbstractVector{<:AbstractFloat}, params::ConstPolyParams)
    return const_polynomial.(x, Ref(params))
end

"""
    background_model(
        x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}, 
        params::BackgroundParams,
    )

Evaluate the combined background (polynomial of 2nd order) at `x`.

Each term is optional — set the corresponding field to `false` in 
[`BackgroundParams`](@ref) to exclude it.

# Arguments
- `x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}`: position(s) at which to
  evaluate in keV
- `params::BackgroundParams`: background parameters

# Returns
- Scalar: the evaluated quadratic background value at `x` in counts/keV
- Vector: an array of evaluated background values at each element of `x` in counts/keV

# Examples

```julia
mu = 2048.0
b = BackgroundParams(linPoly = LinPolyParams(C = 100.0, mu = mu))
background_model(1.0:4096.0, b)
```

# See also
[`BackgroundParams`](@ref) for the parameters.
"""
function background_model(x::AbstractFloat, params::BackgroundParams)
    result = zero(float(x))
    !isa(params.quadPoly, Bool) && (result += quad_polynomial(x, params.quadPoly))
    !isa(params.linPoly, Bool) && (result += lin_polynomial(x, params.linPoly))
    !isa(params.constPoly, Bool) && (result += const_polynomial(x, params.constPoly))
    return result
end
function background_model(x::AbstractVector{<:AbstractFloat}, params::BackgroundParams)
    result = zeros(float(eltype(x)), length(x))
    !isa(params.quadPoly, Bool) && (result .+= quad_polynomial(x, params.quadPoly))
    !isa(params.linPoly, Bool) && (result .+= lin_polynomial(x, params.linPoly))
    !isa(params.constPoly, Bool) && (result .+= const_polynomial(x, params.constPoly))
    return result
end

"""
    full_model(x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}, params::ModelParams)

Evaluate the complete gamma-peak model (peak shape + background) at `x`.

Combines [`peak_model`](@ref) and [`background_model`](@ref). Each term is optional — set
the corresponding field to `false` in [`ModelParams`](@ref) to exclude it, or `true` to
enable it.

# Arguments
- `x::Union{AbstractFloat, AbstractVector{<:AbstractFloat}}`: position(s) at which to 
  evaluate in keV
- `params::ModelParams`: full model parameters

# Returns
- Scalar: the evaluated model value (peak + background) at `x` in counts/keV
- Vector: an array of total model values at each element of `x` in counts/keV

# Examples

```julia
mu = 2048.0
sigma = 10.0
p = PeakParams(gaussian = GaussianParams(A = 100.0, mu = mu, sigma = sigma))
b = BackgroundParams(linPoly = LinPolyParams(C = 100.0, mu = mu))
m = ModelParams(peak = p, background = b)
full_model(1.0:4096.0, m)
```

# See also
- [`ModelParams`](@ref) for the parameters
- [`peak_model`](@ref), and [`background_model`](@ref) for the components
"""
function full_model(x::AbstractFloat, params::ModelParams)
    result = zero(float(x))
    isa(params.peak, PeakParams) && (result += peak_model(x, params.peak))
    isa(params.background, BackgroundParams) &&
        (result += background_model(x, params.background))
    return result
end
function full_model(x::AbstractVector{<:AbstractFloat}, params::ModelParams)
    result = zeros(float(eltype(x)), length(x))
    isa(params.peak, PeakParams) && (result .+= peak_model(x, params.peak))
    isa(params.background, BackgroundParams) &&
        (result .+= background_model(x, params.background))
    return result
end
