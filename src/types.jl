"""
    GaussianParams{T<:AbstractFloat}

Additional parameters for a scaled Gaussian (normal) peak component.

`mu` and `sigma` are usually the same between all model components.

# Fields
- `A::T`: total integrated peak area in counts
- `mu::T`: centroid position of the peak on the x-axis in keV
- `sigma::T`: standard deviation of the Gaussian core (`sigma > 0`) in keV

# Mathematical definition

```math
f(x) = \\frac{A}{\\sigma\\sqrt{2\\pi}} \\,
       \\exp\\!\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)
```

# See also
[`gaussian`](@ref) for evaluating the gaussian.
"""
Base.@kwdef struct GaussianParams{T<:AbstractFloat}
    A::T
    mu::T
    sigma::T
end

"""
    ComptonParams{T<:AbstractFloat}

Parameters for a Compton-edge step function component, modelled as a scaled complementary 
error function.

`mu` and `sigma` are usually the same between all model components.

# Fields
- `h::T`: step height in counts/keV
- `mu::T`: centroid position of the peak on the x-axis in keV
- `sigma::T`: standard deviation of the Gaussian core (`sigma > 0`) in keV

# Mathematical definition

```math
f(x) = \\frac{h}{2} \\,
       \\operatorname{erfc}\\!\\left(\\frac{x-\\mu}{\\sigma\\sqrt{2}}\\right)
```

# See also
[`compton`](@ref) for evaluating the Compton-edge.
"""
Base.@kwdef struct ComptonParams{T<:AbstractFloat}
    h::T
    mu::T
    sigma::T
end

"""
    ExGaussianParams{T<:AbstractFloat}

Parameters for an exponentially modified Gaussian (ex-Gaussian) tail component, used to 
model low- or high-energy tailing in gamma peaks.

`mu` and `sigma` are usually the same between all model components.

# Fields
- `A::T`: total integrated tail area in counts
- `lambda::T`: rate parameter of the exponential tail in 1/keV
- `mu::T`: centroid position of the peak on the x-axis in keV
- `sigma::T`: standard deviation of the Gaussian core (`sigma > 0`) in keV

# Mathematical definition

```math
f(x) = \\frac{A\\lambda}{2} \\,
       \\exp\\!\\left(\\frac{\\lambda}{2}
       (2\\mu + \\lambda\\sigma^2 - 2x)\\right) \\,
       \\operatorname{erfc}\\!\\left(
       \\frac{\\mu + \\lambda\\sigma^2 - x}{\\sigma\\sqrt{2}}\\right)
```

# See also
[`exGaussian`](@ref) for evaluating the tail component.
"""
Base.@kwdef struct ExGaussianParams{T<:AbstractFloat}
    A::T
    lambda::T
    mu::T
    sigma::T
end

"""
    PeakParams

Aggregate container for all components that form a gamma-ray peak.

Each component is optional — set the corresponding field to `false` to exclude it. 
Setting it to `true` instead of specifiying a `Params` object allows for controlling which 
component is used in the fitting process (See [build_prior](@ref)).
`mu` and `sigma` are usually the same between all model components.

# Fields
- `gaussian::Union{GaussianParams,Bool}`: main Gaussian peak shape
- `compton::Union{ComptonParams,Bool}`: Compton-edge step function
- `lowEnergyTail::Union{ExGaussianParams,Bool}`: ex-Gaussian low-energy tail
- `highEnergyTail::Union{ExGaussianParams,Bool}`: ex-Gaussian high-energy tail

# Examples

```julia
# Specify a model with values
mu = 2048.0
sigma = 10.0
p = PeakParams(gaussian = GaussianParams(A = 1000.0, mu = mu, sigma = sigma))

# Enable specific components for later fitting
p = PeakParams(gaussian = true)
```

# See also
- [`peak_model`](@ref) for evaluating the combined peak shape
- [`GaussianParams`](@ref), [`ComptonParams`](@ref), and [`ExGaussianParams`] for the
  component parameters.
"""
Base.@kwdef struct PeakParams
    gaussian::Union{GaussianParams,Bool} = false
    compton::Union{ComptonParams,Bool} = false
    lowEnergyTail::Union{ExGaussianParams,Bool} = false
    highEnergyTail::Union{ExGaussianParams,Bool} = false
end

"""
    QuadPolyParams{T<:AbstractFloat}

Parameters for the quadratic polynomial term in the background model.

# Fields
- `C::T`: scaling coefficient in counts/keV³
- `mu::T`: centering value for the polynomial expansion in keV. Usually the centroid of the
  gamma-peak.

# Mathematical definition

```math
f(x) = C \\cdot (x - \\mu)^2
```

# See also
[`quad_polynomial`](@ref) for evaluating the term.
"""
Base.@kwdef struct QuadPolyParams{T<:AbstractFloat}
    C::T
    mu::T
end

"""
    LinPolyParams{T<:AbstractFloat}

Parameters for the linear polynomial term in the background model.

# Fields
- `C::T`: scaling coefficient in counts/keV²
- `mu::T`: centering value for the polynomial expansion in keV. Usually the centroid of the
  gamma-peak.

# Mathematical definition

```math
f(x) = C \\cdot (x - \\mu)
```

# See also
[`lin_polynomial`](@ref) for evaluating the term.
"""
Base.@kwdef struct LinPolyParams{T<:AbstractFloat}
    C::T
    mu::T
end

"""
    ConstPolyParams{T<:AbstractFloat}

Parameters for the constant polynomial term in the background model.

# Fields
- `C::T`: constant offset in counts/keV

# See also
[`const_polynomial`](@ref) for evaluating the term.
"""
Base.@kwdef struct ConstPolyParams{T<:AbstractFloat}
    C::T
end

"""
    BackgroundParams

Aggregate container for all components that form the background model.

Each component is optional — set the corresponding field to `false` to exclude it. 
Setting it to `true` instead of specifiying a `Params` object allows for controlling which 
component is used in the fitting process (See [build_prior](@ref)).
`mu` is usually the same between all model components.

# Fields
- `quadPoly::Union{QuadPolyParams,Bool}`: quadratic polynomial term
- `linPoly::Union{LinPolyParams,Bool}`: linear polynomial term
- `constPoly::Union{ConstPolyParams,Bool}`: constant polynomial term

# Examples

```julia
# Specify a model with values
mu = 2048.0
b = BackgroundParams(linPoly = LinPolyParams(C = 1.0, mu = mu))

# Enable specific components for later fitting
b = BackgroundParams(linPoly = true)
```

# See also
[background_model](@ref) for evaluating the background model.
"""
Base.@kwdef struct BackgroundParams
    quadPoly::Union{QuadPolyParams,Bool} = false
    linPoly::Union{LinPolyParams,Bool} = false
    constPoly::Union{ConstPolyParams,Bool} = false
end

"""
    ModelParams

Complete model combining a gamma peak and a polynomial background.

Each component is optional — unset fields are `nothing` and are skipped during evaluation.
`mu` and `sigma` are usually the same between all model components.

# Fields
- `peak::Union{PeakParams,Nothing}`: peak shape parameters (Gaussian, Compton edge, 
  tails)
- `background::Union{BackgroundParams,Nothing}`: quadratic background parameters

# Examples

```julia
# Specify a model with values
mu = 2048.0
sigma = 10.0
p = PeakParams(gaussian = GaussianParams(A = 100.0, mu = mu, sigma = sigma))
b = BackgroundParams(linPoly = LinPolyParams(C = 1.0, mu = mu))
m = ModelParams(peak = p, background = b)

# Enable specific components for later fitting
p = PeakParams(gaussian = true)
b = BackgroundParams(linPoly = true)
m = ModelParams(peak = p, background = b)
```

# See also
- [`full_model`](@ref) for evaluating the combined model
- [`PeakParams`](@ref), and [`BackgroundParams`](@ref) for the component parameters
"""
Base.@kwdef struct ModelParams
    peak::Union{PeakParams,Nothing} = nothing
    background::Union{BackgroundParams,Nothing} = nothing
end

"""
    SpectrumData{T<:AbstractFloat, U<:Integer}

Container for binned energy spectrum data.

# Fields
- `bin_centers::AbstractVector{T}`: bin center(s) in keV
- `weights::AbstractVector{U}`: observed count(s) per bin
- `bin_size::T`: width of each bin in keV
"""
Base.@kwdef struct SpectrumData{T<:AbstractFloat, U<:Integer}
    bin_centers::AbstractVector{T}
    weights::AbstractVector{U}
    bin_size::T
end
