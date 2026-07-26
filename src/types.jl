"""
    GaussianParams{T<:Real}

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
Base.@kwdef struct GaussianParams{T<:Real}
    A::T
    mu::T
    sigma::T
end

"""
    ComptonParams{T<:Real}

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
Base.@kwdef struct ComptonParams{T<:Real}
    h::T
    mu::T
    sigma::T
end

"""
    ExGaussianParams{T<:Real}

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
Base.@kwdef struct ExGaussianParams{T<:Real}
    A::T
    lambda::T
    mu::T
    sigma::T
end

"""
    BackgroundParams{T<:Real}

Parameters for a quadratic polynomial background model, centered at `mu` to improve
numerical stability during fitting.

Each term is optional — unset fields are `nothing` and are skipped during evaluation.
`mu` is usually the centroid of the gamma-peak and the same between all model components.

# Fields
- `b2::Union{T,Nothing}`: quadratic coefficient (``(x - \\mu)^2`` term) in counts/keV³
- `b1::Union{T,Nothing}`: linear coefficient (``(x - \\mu)`` term) in counts/keV²
- `b0::Union{T,Nothing}`: constant offset in counts/keV
- `mu::T`: centering value for the polynomial expansion in keV. Usually the centroid of the 
  gamma-peak

# Mathematical definition

```math
f(x) = b_2 (x - \\mu)^2 + b_1 (x - \\mu) + b_0
```

# See also
[background_model](@ref) for evaluating the background model.
"""
Base.@kwdef struct BackgroundParams{T<:Real}
    b2::Union{T,Nothing} = nothing
    b1::Union{T,Nothing} = nothing
    b0::Union{T,Nothing} = nothing
    mu::T
end

"""
    PeakParams{T<:Real}

Aggregate container for all components that form a gamma-ray peak.

Each component is optional — unset fields are `nothing` and are skipped during evaluation.
`mu` and `sigma` are usually the same between all model components.

# Fields
- `gaussian::Union{GaussianParams{T},Nothing}`: main Gaussian peak shape
- `compton::Union{ComptonParams{T},Nothing}`: Compton-edge step function
- `lowEnergyTail::Union{ExGaussianParams{T},Nothing}`: ex-Gaussian low-energy tail
- `highEnergyTail::Union{ExGaussianParams{T},Nothing}`: ex-Gaussian high-energy tail

# See also
- [`peak_model`](@ref) for evaluating the combined peak shape
- [`GaussianParams`](@ref), [`ComptonParams`](@ref), and [`ExGaussianParams`] for the
  component parameters.
"""
Base.@kwdef struct PeakParams{T<:Real}
    gaussian::Union{GaussianParams{T},Nothing} = nothing
    compton::Union{ComptonParams{T},Nothing} = nothing
    lowEnergyTail::Union{ExGaussianParams{T},Nothing} = nothing
    highEnergyTail::Union{ExGaussianParams{T},Nothing} = nothing
end

"""
    ModelParams{T<:Real}

Complete model combining a gamma peak and a polynomial background.

Each component is optional — unset fields are `nothing` and are skipped during evaluation.
`mu` and `sigma` are usually the same between all model components.

# Fields
- `peak::Union{PeakParams{T},Nothing}`: peak shape parameters (Gaussian, Compton edge, 
  tails)
- `background::Union{BackgroundParams{T},Nothing}`: quadratic background parameters

```julia
mu = 2048.0
sigma = 10.0
p = PeakParams(gaussian = GaussianParams(A = 100.0, mu = mu, sigma = sigma))
b = BackgroundParams(b0 = 100.0, mu = mu)
m = ModelParams(peak = p, background = b)
```

# See also
- [`full_model`](@ref) for evaluating the combined model
- [`PeakParams`](@ref), and [`BackgroundParams`](@ref) for the component parameters
"""
Base.@kwdef struct ModelParams{T<:Real}
    peak::Union{PeakParams{T},Nothing} = nothing
    background::Union{BackgroundParams{T},Nothing} = nothing
end
