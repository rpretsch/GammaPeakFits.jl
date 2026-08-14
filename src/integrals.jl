"""
    numerical_integral(data::SpectrumData, params::ModelParams)

Integrate the [`full_model`](@ref) numerically over each energy bin of size `bin_size` 
using `QuadGK.quadgk`.

Components marked `Disabled()` contribute zero.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::ModelParams`: model parameters

# Returns
- An array of expected counts per bin

# See also
- [`full_model`](@ref) for the integrated model
- [`ModelParams`](@ref) for the model params
- [`Enabled`](@ref), [`Disabled`](@ref), [`AbstractComponent`](@ref) for the component 
  management
"""
function numerical_integral(data::SpectrumData, params::ModelParams)
    return first.(
        quadgk.(
            x -> full_model(x, params),
            data.bin_edges[1:(end-1)],
            data.bin_edges[2:end],
        ),
    )
end

"""
    midpoint_integral(data::SpectrumData, params::ModelParams)

Integrate the [`full_model`](@ref) via the midpoint rule over each energy bin of size 
`bin_size`.

Components marked `Disabled()` contribute zero.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::ModelParams`: model parameters

# Returns
- An array of expected counts per bin

# See also
- [`full_model`](@ref) for the integrated model
- [`ModelParams`](@ref) for the model params
- [`Enabled`](@ref), [`Disabled`](@ref), [`AbstractComponent`](@ref) for the component 
  management
"""
function midpoint_integral(data::SpectrumData, params::ModelParams)
    return full_model(data.bin_centers, params) .* data.bin_size
end

"""
    analytical_integral(data::SpectrumData, params::ModelParams)

Integrate the [`full_model`](@ref) analytically over each energy bin of size `bin_size`.

Components set to `Disabled()` contribute zero.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::ModelParams`: model parameters

# Returns
- An array of expected counts per bin

# See also
- [`gaussian_integral`](@ref), [`compton_integral`](@ref), [`exGaussian_integral`](@ref),
  [`quadPoly_integral`](@ref), [`linPoly_integral`](@ref), and [`constPoly_integral`](@ref)
  for the individual components
- [`full_model`](@ref) for the integrated model
- [`ModelParams`](@ref) for the model params
- [`Enabled`](@ref), [`Disabled`](@ref), [`AbstractComponent`](@ref) for the component 
  management
"""
function analytical_integral(data::SpectrumData, params::ModelParams)
    return _integral(data, params.peak) .+ _integral(data, params.background)
end

"""
    _integral(data::SpectrumData, params::AbstractComponent)

Integrate a model component (or container) analytically over each energy bin, dispatching 
on the component type.

Components marked `Disabled()` contribute zero.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::AbstractComponent`: component parameters (or container)

# Returns
- An array of expected counts per bin for the given component

# See also
- [`analytical_integral`](@ref) for the combined integral
- [`_expected_counts`](@ref) for the integration-method dispatch
"""
# Disabled components
_integral(data::SpectrumData, ::Disabled) =
    zeros(eltype(data.bin_centers), length(data.bin_centers))

# Peak components
_integral(data::SpectrumData, params::GaussianParams) = gaussian_integral(data, params)
_integral(data::SpectrumData, params::ComptonParams) = compton_integral(data, params)
_integral(data::SpectrumData, params::ExGaussianParams) = exGaussian_integral(data, params)

# Background components
_integral(data::SpectrumData, params::QuadPolyParams) = quadPoly_integral(data, params)
_integral(data::SpectrumData, params::LinPolyParams) = linPoly_integral(data, params)
_integral(data::SpectrumData, params::ConstPolyParams) = constPoly_integral(data, params)

# Model components
_integral(data::SpectrumData, params::PeakParams) =
    _integral(data, params.gaussian) .+ _integral(data, params.compton) .+
    _integral(data, params.lowEnergyTail) .+ _integral(data, params.highEnergyTail)
_integral(data::SpectrumData, params::BackgroundParams) =
    _integral(data, params.quadPoly) .+ _integral(data, params.linPoly) .+
    _integral(data, params.constPoly)

"""
    gaussian_integral(data::SpectrumData, params::GaussianParams)

Integrate the scaled Gaussian component analytically over each energy bin.

Uses `SpecialFunctions.erf`.

# Mathematical definition

```math
F(x) = \\frac{A}{2}\\,
       \\text{erf}\\!\\left(\\frac{x-\\mu}{\\sqrt{2}\\sigma}\\right)
```

The count in a bin is `F(e_{i+1}) - F(e_i)` at the stored bin edges ``e``.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::GaussianParams`: Gaussian component parameters

# Returns
- An array of expected Gaussian counts per bin

# See also
- [`gaussian`](@ref) for the component model
- [`GaussianParams`](@ref) for the parameters
"""
function gaussian_integral(data::SpectrumData, params::GaussianParams)
    function _antiderivative(x::Float64, params::GaussianParams)
        return params.A/2 * erf((x - params.mu)/(sqrt(2) * params.sigma))
    end
    antiderivative_values = _antiderivative.(data.bin_edges, Ref(params))
    return antiderivative_values[2:end] .- antiderivative_values[1:(end-1)]
end

"""
    compton_integral(data::SpectrumData, params::ComptonParams)

Integrate the Compton-edge step function component analytically over each energy bin.

# Mathematical definition

```math
F(x) = (x-\\mu)\\,f(x) - \\frac{h\\sigma}{\\sqrt{2\\pi}}\\,
       \\exp\\!\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)
```

The count in a bin is `F(e_{i+1}) - F(e_i)` at the stored bin edges ``e``, and ``f(x)`` 
being the model component itself.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::ComptonParams`: Compton component parameters

# Returns
- An array of expected Compton counts per bin

# See also
- [`compton`](@ref) for the component model
- [`ComptonParams`](@ref) for the parameters
"""
function compton_integral(data::SpectrumData, params::ComptonParams)
    function _antiderivative(x::Float64, params::ComptonParams)
        return (x - params.mu) * compton(x, params) -
               params.h * params.sigma / sqrt(2 * pi) *
               exp(-(x - params.mu)^2 / (2 * params.sigma^2))
    end
    antiderivative_values = _antiderivative.(data.bin_edges, Ref(params))
    return antiderivative_values[2:end] .- antiderivative_values[1:(end-1)]
end

"""
    exGaussian_integral(data::SpectrumData, params::ExGaussianParams)

Integrate the ex-Gaussian tail component analytically over each energy bin.

# Mathematical definition

```math
F(x) = \\frac{A}{2}\\,\\text{erf}\\!\\left(\\frac{x-\\mu}{\\sqrt{2}\\sigma}\\right)
       \\pm\\tau f(x)
```

The count in a bin is `F(e_{i+1}) - F(e_i)` at the stored bin edges ``e``, and ``f(x)`` 
being the model component itself. 
The low-/high-energy tails correspond to a ``+``/``-`` sign for the ``\\pm`` sign above,
respectively.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::ExGaussianParams`: tail component parameters

# Returns
- An array of expected tail counts per bin

# See also
- [`exGaussian`](@ref) for the component model
- [`ExGaussianParams`](@ref) for the parameters
"""
function exGaussian_integral(data::SpectrumData, params::ExGaussianParams)
    function _antiderivative(x::Float64, params::ExGaussianParams)
        return params.A/2 * erf((x - params.mu)/(sqrt(2) * params.sigma)) -
               (-1)^params.is_lowEnergyTail * params.tau * exGaussian(x, params)
    end
    antiderivative_values = _antiderivative.(data.bin_edges, Ref(params))
    return antiderivative_values[2:end] .- antiderivative_values[1:(end-1)]
end

"""
    quadPoly_integral(data::SpectrumData, params::QuadPolyParams)

Integrate the quadratic polynomial component analytically over each energy bin.

# Mathematical definition

```math
F(x) = \\frac{C}{3}\\,(x-\\mu)^3
```

The count in a bin is `F(e_{i+1}) - F(e_i)` at the stored bin edges ``e``.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::QuadPolyParams`: quadratic polynomial parameters

# Returns
- An array of expected quadratic background counts per bin

# See also
- [`quad_polynomial`](@ref) for the component model
- [`QuadPolyParams`](@ref) for the parameters
"""
function quadPoly_integral(data::SpectrumData, params::QuadPolyParams)
    function _antiderivative(x::Float64, params::QuadPolyParams)
        return params.C/3 * (x - params.mu)^3
    end
    antiderivative_values = _antiderivative.(data.bin_edges, Ref(params))
    return antiderivative_values[2:end] .- antiderivative_values[1:(end-1)]
end

"""
    linPoly_integral(data::SpectrumData, params::LinPolyParams)

Integrate the linear polynomial component analytically over each energy bin.

# Mathematical definition

```math
F(x) = \\frac{C}{2}\\,x\\,(x - 2\\mu)
```

The count in a bin is `F(e_{i+1}) - F(e_i)` at the stored bin edges ``e``.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::LinPolyParams`: linear polynomial parameters

# Returns
- An array of expected linear background counts per bin

# See also
- [`lin_polynomial`](@ref) for the component model
- [`LinPolyParams`](@ref) for the parameters
"""
function linPoly_integral(data::SpectrumData, params::LinPolyParams)
    function _antiderivative(x::Float64, params::LinPolyParams)
        return params.C/2 * x * (x - 2 * params.mu)
    end
    antiderivative_values = _antiderivative.(data.bin_edges, Ref(params))
    return antiderivative_values[2:end] .- antiderivative_values[1:(end-1)]
end

"""
    constPoly_integral(data::SpectrumData, params::ConstPolyParams)

Integrate the constant polynomial component analytically over each energy bin.

# Mathematical definition

```math
F(x) = C\\,x
```

The count in a bin is `F(e_{i+1}) - F(e_i)` at the stored bin edges ``e``.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::ConstPolyParams`: constant polynomial parameters

# Returns
- An array of expected constant background counts per bin

# See also
- [`const_polynomial`](@ref) for the component model
- [`ConstPolyParams`](@ref) for the parameters
"""
function constPoly_integral(data::SpectrumData, params::ConstPolyParams)
    function _antiderivative(x::Float64, params::ConstPolyParams)
        return params.C * x
    end
    antiderivative_values = _antiderivative.(data.bin_edges, Ref(params))
    return antiderivative_values[2:end] .- antiderivative_values[1:(end-1)]
end
