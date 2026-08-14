"""
    gaussian(x::Union{Float64,Vector{Float64}}, params::GaussianParams)

Evaluate a scaled Gaussian (normal) distribution at `x`.

Uses `Distributions.jl` for the normal distribution implementation.

# Mathematical definition

```math
f(x) = \\frac{A}{\\sqrt{2\\pi}\\sigma} \\,
       \\exp\\!\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)
```

# Arguments
- `x::Union{Float64,Vector{Float64}}`: position(s) at which to evaluate in keV
- `params::GaussianParams`: component parameters

# Throws
- An `ArgumentError` if `params.sigma` is negative

# Returns
- Scalar: the evaluated Gaussian amplitude at `x` in counts/keV
- Vector: an array of evaluated Gaussian amplitudes at each element of `x` counts/keV

# See also
- [`GaussianParams`](@ref) for the parameters
"""
function gaussian(x::Float64, params::GaussianParams)
    sigma = params.sigma
    sigma <= 0 && throw(ArgumentError("`sigma` can't be negative"))
    return params.A * pdf(Normal(params.mu, sigma), x)
end
function gaussian(x::Vector{Float64}, params::GaussianParams)
    return gaussian.(x, Ref(params))
end

"""
    compton(x::Union{Float64,Vector{Float64}}, params::ComptonParams)

Evaluate a Compton-edge step function component at `x`.

Uses the complementary error function (`erfc`) from `SpecialFunctions.jl` to model the
smooth step from Compton scattering.

# Mathematical definition

```math
f(x) = \\frac{h}{2}\\,
       \\text{erfc}\\!\\left(\\frac{x-\\mu}{\\sigma\\sqrt{2}}\\right)
```

# Arguments
- `x::Union{Float64,Vector{Float64}}`: position(s) at which to evaluate in keV
- `params::ComptonParams`: component parameters

# Returns
- Scalar: the evaluated Compton step height at `x` in counts/keV
- Vector: an array of evaluated Compton step heights at each element of `x` in counts/keV

# Throws
- An `ArgumentError` if `params.sigma` is negative or zero

# See also
- [`ComptonParams`](@ref) for the parameters
"""
function compton(x::Float64, params::ComptonParams)
    sigma = params.sigma
    sigma <= 0 && throw(ArgumentError("`sigma` can't be zero or negative"))
    return params.h/2 * erfc((x - params.mu)/(sqrt(2) * params.sigma))
end
function compton(x::Vector{Float64}, params::ComptonParams)
    return compton.(x, Ref(params))
end

"""
    exGaussian(x::Union{Float64,Vector{Float64}}, params::ExGaussianParams)

Evaluate an exponentially modified Gaussian (ex-Gaussian) tail component at `x`.

Used to model asymmetric peak tailing.
Is evaluated in log-space using the `SpecialFunctions.logerfcx` for numerical stability
reasons.

# Mathematical definition

```math
f(x) = \\frac{A}{2\\tau}\\,
       \\exp\\!\\left(\\frac{1}{2}\\left(\\frac{\\sigma}{\\tau}\\right)^2
       \\pm\\frac{x-\\mu}{\\tau}\\right)\\,
       \\text{erfc}\\!\\left(\\frac{1}{\\sqrt{2}}\\left(\\frac{\\sigma}{\\tau}
       \\pm\\frac{x-\\mu}{\\sigma}\\right)\\right)
```

The low-/high-energy tails correspond to a ``+``/``-`` sign for the ``\\pm`` sign above,
respectively. 

For numerical stability this is evaluated in log-space as

```math
\\log f(x) = \\log A - \\log(2\\tau) 
             -\\frac{1}{2} \\left(\\frac{x-\\mu}{\\sigma}\\right)^2 
             +\\text{logerfcx}\\!\\left(\\frac{1}{\\sqrt{2}}\\left(\\frac{\\sigma}{\\tau}
             \\pm\\frac{x-\\mu}{\\sigma}\\right)\\right)
```

# Arguments
- `x::Union{Float64,Vector{Float64}}`: position(s) at which to evaluate in keV
- `params::ExGaussianParams`: component parameters

# Returns
- Scalar: the evaluated ex-Gaussian tail amplitude at `x` in counts/keV
- Vector: an array of evaluated tail amplitudes at each element of `x` in counts/keV

# Throws
- An `ArgumentError` if either `params.sigma` and `params.tau` are negative or zero

# See also
- [`ExGaussianParams`](@ref) for the parameters
"""
function exGaussian(x::Float64, params::ExGaussianParams)
    tau = params.tau
    tau <= 0 && throw(ArgumentError("`tau` can't be zero or negative"))
    sigma = params.sigma
    sigma <= 0 && throw(ArgumentError("`sigma` can't be zero or negative"))
    mu = params.mu

    logf =
        log(params.A) - log(2 * tau) - 1/2 * ((x - mu)/sigma)^2 +
        logerfcx(1/sqrt(2) * (sigma/tau - (-1)^params.is_lowEnergyTail * (x - mu)/sigma))
    return exp(logf)
end
function exGaussian(x::Vector{Float64}, params::ExGaussianParams)
    return exGaussian.(x, Ref(params))
end

"""
    peak_model(x::Union{Float64,Vector{Float64}}, params::PeakParams)

Evaluate the combined peak shape (Gaussian + Compton edge + ex-Gaussian tails) at `x`.

Each term is optional — set the corresponding field to `Disabled()` in [`PeakParams`](@ref) 
to exclude it.

# Mathematical definition

```math
f_{\\text{peak}}(x) =
    f_{\\text{G}}(x) + f_{\\text{C}}(x) + f_{\\text{eG,low}}(x) + f_{\\text{eG,high}}(x)
```

where each term is optional.

# Arguments
- `x::Union{Float64,Vector{Float64}}`: position(s) at which to evaluate in keV
- `params::PeakParams`: peak component parameters

# Returns
- Scalar: the summed peak amplitude at `x` in counts/keV
- Vector: an array of summed peak amplitudes at each element of `x` in counts/keV

# See also
- [`PeakParams`](@ref) for the parameters
- [`gaussian`](@ref), [`compton`](@ref), and [`exGaussian`](@ref) for the components
"""
function peak_model(x::Union{Float64,Vector{Float64}}, params::PeakParams)
    return _value(x, params.gaussian) .+ _value(x, params.compton) .+
           _value(x, params.lowEnergyTail) .+ _value(x, params.highEnergyTail)
end

"""
    quad_polynomial(x::Union{Float64,Vector{Float64}}, params::QuadPolyParams)

Evaluate a scaled quadratic polynomial term at `x`.

# Mathematical definition

```math
f(x) = C \\cdot (x - \\mu)^2
```

# Arguments
- `x::Union{Float64,Vector{Float64}}`: position(s) at which to evaluate in keV
- `params::QuadPolyParams`: component parameters

# Returns
- Scalar: the evaluated quadratic polynomial amplitude at `x` in counts/keV
- Vector: an array of evaluated quadratic polynomial amplitudes at each element of `x` in 
  counts/keV

# See also
- [`QuadPolyParams`](@ref) for the parameter structure
"""
function quad_polynomial(x::Float64, params::QuadPolyParams)
    return params.C * (x - params.mu)^2
end
function quad_polynomial(x::Vector{Float64}, params::QuadPolyParams)
    return quad_polynomial.(x, Ref(params))
end

"""
    lin_polynomial(x::Union{Float64,Vector{Float64}}, params::LinPolyParams)

Evaluate a scaled linear polynomial term at `x`.

# Mathematical definition

```math
f(x) = C \\cdot (x - \\mu)
```

# Arguments
- `x::Union{Float64,Vector{Float64}}`: position(s) at which to evaluate in keV
- `params::LinPolyParams`: component parameters

# Returns
- Scalar: the evaluated linear polynomial amplitude at `x` in counts/keV
- Vector: an array of evaluated linear polynomial amplitudes at each element of `x` in 
  counts/keV

# See also
- [`LinPolyParams`](@ref) for the parameter structure
"""
function lin_polynomial(x::Float64, params::LinPolyParams)
    return params.C * (x - params.mu)
end
function lin_polynomial(x::Vector{Float64}, params::LinPolyParams)
    return lin_polynomial.(x, Ref(params))
end

"""
    const_polynomial(
        x::Union{Float64,Vector{Float64}}, 
        params::ConstPolyParams,
    )

Evaluate a constant polynomial term.

Uses `x` to set whether to return as a scalar or vector.

# Mathematical definition

```math
f(x) = C
```

# Arguments
- `x::Union{Float64,Vector{Float64}}`: sets the return type (scalar vs vector)
- `params::ConstPolyParams`: polynomial parameters

# Returns
- Scalar: the constant coefficient in counts/keV
- Vector: an array of constant coefficient in counts/keV

# See also
- [`ConstPolyParams`](@ref) for the parameter structure
"""
function const_polynomial(x::Float64, params::ConstPolyParams)
    return params.C
end
function const_polynomial(x::Vector{Float64}, params::ConstPolyParams)
    return const_polynomial.(x, Ref(params))
end

"""
    background_model(x::Union{Float64,Vector{Float64}}, params::BackgroundParams)

Evaluate the combined background (polynomial of 2nd order) at `x`.

Each term is optional — set the corresponding field to `Disabled()` in 
[`BackgroundParams`](@ref) to exclude it.

# Mathematical definition

```math
f_{\\text{bg}}(x) =
    C_q\\,(x - \\mu_q)^2 + C_l\\,(x - \\mu_l) + C_c
```

where each term is optional.

# Arguments
- `x::Union{Float64,Vector{Float64}}`: position(s) at which to evaluate in keV
- `params::BackgroundParams`: background parameters

# Returns
- Scalar: the evaluated quadratic background value at `x` in counts/keV
- Vector: an array of evaluated background values at each element of `x` in counts/keV

# See also
- [`BackgroundParams`](@ref) for the parameters
"""
function background_model(x::Union{Float64,Vector{Float64}}, params::BackgroundParams)
    return _value(x, params.quadPoly) .+ _value(x, params.linPoly) .+
           _value(x, params.constPoly)
end

"""
    full_model(x::Union{Float64,Vector{Float64}}, params::ModelParams)

Evaluate the complete gamma-peak model (peak shape + background) at `x`.

Combines [`peak_model`](@ref) and [`background_model`](@ref). Each block is optional — set 
the corresponding field to `Disabled()` in [`ModelParams`](@ref) to exclude it.

# Mathematical definition

```math
f(x) = f_{\\text{peak}}(x) + f_{\\text{bg}}(x)
```

# Arguments
- `x::Union{Float64,Vector{Float64}}`: position(s) at which to evaluate in keV
- `params::ModelParams`: full model parameters

# Returns
- Scalar: the evaluated model value (peak + background) at `x` in counts/keV
- Vector: an array of total model values at each element of `x` in counts/keV

# See also
- [`ModelParams`](@ref) for the parameters
- [`peak_model`](@ref), and [`background_model`](@ref) for the components
"""
function full_model(x::Union{Float64,Vector{Float64}}, params::ModelParams)
    return _value(x, params.peak) .+ _value(x, params.background)
end

"""
    _value(x::Union{Float64,Vector{Float64}}, params::AbstractComponent)

Evaluate a model component (or container) at `x`, dispatching on the component type.

Components marked `Disabled()` contribute zero.

# Arguments
- `x::Union{Float64,Vector{Float64}}`: position(s) at which to 
  evaluate in keV
- `params::AbstractComponent`: component parameters (or container)

# Returns
- Scalar: the evaluated component value at `x` in counts/keV
- Vector: an array of evaluated component values at each element of `x` in counts/keV

# See also
- [`full_model`](@ref) for the combined model
"""
# Disabled components
_value(x::Float64, ::Disabled) = zero(x)
_value(x::Vector{Float64}, ::Disabled) = zeros(eltype(x), length(x))

# Peak components
_value(x::Union{Float64,Vector{Float64}}, params::GaussianParams) = gaussian(x, params)
_value(x::Union{Float64,Vector{Float64}}, params::ComptonParams) = compton(x, params)
_value(x::Union{Float64,Vector{Float64}}, params::ExGaussianParams) = exGaussian(x, params)

# Background components
_value(x::Union{Float64,Vector{Float64}}, params::QuadPolyParams) =
    quad_polynomial(x, params)
_value(x::Union{Float64,Vector{Float64}}, params::LinPolyParams) = lin_polynomial(x, params)
_value(x::Union{Float64,Vector{Float64}}, params::ConstPolyParams) =
    const_polynomial(x, params)

# Model components
_value(x::Union{Float64,Vector{Float64}}, params::PeakParams) = peak_model(x, params)
_value(x::Union{Float64,Vector{Float64}}, params::BackgroundParams) =
    background_model(x, params)
