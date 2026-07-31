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
f(x) = \\frac{A}{\\sqrt{2\\pi}\\sigma} \\,
       \\exp\\!\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)
```

# See also
- [`gaussian`](@ref) for evaluating the gaussian
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
- [`compton`](@ref) for evaluating the Compton-edge
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
- `tau::T`: Exponent relaxation time of the exponential tail in keV
- `is_lowEnergyTail::Bool`: Tail direction (`true`/`false` for low-/high-energy tails  
  respectively)
- `mu::T`: centroid position of the Gaussian core on the x-axis in keV
- `sigma::T`: standard deviation of the Gaussian core in keV

# Mathematical definition

```math
f(x) = \\frac{A}{2\\tau}\\,
       \\exp\\!\\left(-\\frac{1}{2}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2\\right)\\,
       \\text{erfcx}\\!\\left(\\frac{1}{\\sqrt{2}}\\left(\\frac{\\sigma}{\\tau}
       \\pm\\cdot\\frac{x-\\mu}{\\sigma}\\right)\\right)
```

The low-/high-energy tails correspond to a `-`/`+` sign for the `±` sign above respectively.

# See also
- [`exGaussian`](@ref) for evaluating the tail component
"""
Base.@kwdef struct ExGaussianParams{T<:AbstractFloat}
    A::T
    tau::T
    is_lowEnergyTail::Bool
    mu::T
    sigma::T
end

"""
    PeakParams

Aggregate container for all components that form a gamma-ray peak.

Each component is optional — set the corresponding field to `false` to exclude it. 
Setting it to `true` instead of specifying a `Params` object allows for controlling which 
component is used in the fitting process (See [build_prior](@ref)).
`mu` and `sigma` are usually the same between all model components.

# Fields
- `gaussian::Union{GaussianParams,Bool}`: main Gaussian peak shape
- `compton::Union{ComptonParams,Bool}`: Compton-edge step function
- `lowEnergyTail::Union{ExGaussianParams,Bool}`: ex-Gaussian low-energy tail
- `highEnergyTail::Union{ExGaussianParams,Bool}`: ex-Gaussian high-energy tail

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
- [`quad_polynomial`](@ref) for evaluating the term
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
- [`lin_polynomial`](@ref) for evaluating the term
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
- [`const_polynomial`](@ref) for evaluating the term
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

# See also
- [background_model](@ref) for evaluating the background model
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

# Constructors

    SpectrumData(
        lower_limit::T, 
        upper_limit::T, 
        bin_size::T, 
        params::ModelParams
    ) {T<:AbstractFloat}

Generate synthetic spectrum data from a model over a uniform grid of 
`(lower_limit):bin_size:(upper_limit)`, then sampling Poisson-distributed counts for each 
bin.

# Arguments
- `lower_limit::T`: start of the energy range in keV
- `upper_limit::T`: end of the energy range in keV
- `bin_size::T`: width of each bin in keV
- `params::ModelParams`: model parameters used to compute expected counts

# Returns
- A `SpectrumData` object

# See also
- [`full_model`](@ref) for the used model
"""
Base.@kwdef struct SpectrumData{T<:AbstractFloat,U<:Integer}
    bin_centers::AbstractVector{T}
    weights::AbstractVector{U}
    bin_size::T
end

function SpectrumData(
    lower_limit::T,
    upper_limit::T,
    bin_size::T,
    params::ModelParams,
) where {T<:AbstractFloat}
    bin_centers = range(lower_limit, upper_limit; step = bin_size)
    expected_counts = full_model(bin_centers, params)
    any(expected_counts .< 0) &&
        throw(ArgumentError("Model produced negative expected counts; check parameters."))
    weights = rand.(Poisson.(expected_counts))
    return SpectrumData(bin_centers = bin_centers, weights = weights, bin_size = bin_size)
end
