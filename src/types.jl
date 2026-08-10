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
       \\text{erfc}\\!\\left(\\frac{x-\\mu}{\\sigma\\sqrt{2}}\\right)
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
- `is_lowEnergyTail::Bool`: Tail direction (`true`/`false` for low-/high-energy tails,
  respectively)
- `mu::T`: centroid position of the Gaussian core on the x-axis in keV
- `sigma::T`: standard deviation of the Gaussian core in keV

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

using `SpecialFunctions.logerfcx`. 

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
- `gaussian::Union{GaussianParams,Bool}`: main Gaussian peak shape. Default: `false`
- `compton::Union{ComptonParams,Bool}`: Compton-edge step function. Default: `false`
- `lowEnergyTail::Union{ExGaussianParams,Bool}`: ex-Gaussian low-energy tail. 
  Default: `false`
- `highEnergyTail::Union{ExGaussianParams,Bool}`: ex-Gaussian high-energy tail.
  Default: `false`

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
Setting it to `true` instead of specifying a `Params` object allows for controlling which 
component is used in the fitting process (See [build_prior](@ref)).
`mu` is usually the same between all model components.

# Fields
- `quadPoly::Union{QuadPolyParams,Bool}`: quadratic polynomial term. Default: `false`
- `linPoly::Union{LinPolyParams,Bool}`: linear polynomial term. Default: `false`
- `constPoly::Union{ConstPolyParams,Bool}`: constant polynomial term. Default: `false`

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
  tails). Default: `nothing`
- `background::Union{BackgroundParams,Nothing}`: quadratic background parameters. 
  Default: `nothing`

# Constructors

    ModelParams(params::NamedTuple)

Build a concrete `ModelParams` from a flat `NamedTuple` of component parameter values (such 
as a sample drawn from the distribution returned by [`build_prior`](@ref)).

A component is enabled or disabled based on the presence of its keys:

| Component | Required keys |
| --- | --- |
| `peak.gaussian` | `:gaussian_A` |
| `peak.compton` | `:compton_h` |
| `peak.lowEnergyTail` | `:lowEnergyTail_A`, `:lowEnergyTail_tau` |
| `peak.highEnergyTail` | `:highEnergyTail_A`, `:highEnergyTail_tau` |
| `background.quadPoly` | `:quadPoly_C` |
| `background.linPoly` | `:linPoly_C` |
| `background.constPoly` | `:constPoly_C` |

`:mu` is shared by all components except `background.constPoly` and is required whenever 
any peak or non-constant background component is present. `:sigma` is shared by all peak
components and is required whenever any peak component is present.

# Throws
- An `ArgumentError` if only one key of the `:lowEnergyTail_A`/`:lowEnergyTail_tau` or
  `:highEnergyTail_A`/`:highEnergyTail_tau` pair is provided
- An `ArgumentError` if `:mu` is missing while a peak or non-constant background component
  is present
- An `ArgumentError` if `:sigma` is missing while a peak component is present

# See also
- [`full_model`](@ref) for evaluating the combined model
- [`PeakParams`](@ref), and [`BackgroundParams`](@ref) for the component parameters
- [`build_prior`](@ref) for constructing the `NamedTuple` distribution this constructor
  consumes
"""
Base.@kwdef struct ModelParams
    peak::Union{PeakParams,Nothing} = nothing
    background::Union{BackgroundParams,Nothing} = nothing
end

function ModelParams(params::NamedTuple)
    has_gaussian = hasproperty(params, :gaussian_A)
    has_compton = hasproperty(params, :compton_h)
    has_lowEnergyTail =
        hasproperty(params, :lowEnergyTail_tau) && hasproperty(params, :lowEnergyTail_A)
    has_highEnergyTail =
        hasproperty(params, :highEnergyTail_tau) && hasproperty(params, :highEnergyTail_A)
    has_quadPoly = hasproperty(params, :quadPoly_C)
    has_linPoly = hasproperty(params, :linPoly_C)
    has_constPoly = hasproperty(params, :constPoly_C)

    has_peak = has_gaussian || has_compton || has_lowEnergyTail || has_highEnergyTail
    has_background = has_quadPoly || has_linPoly || has_constPoly

    xor(hasproperty(params, :lowEnergyTail_A), hasproperty(params, :lowEnergyTail_tau)) &&
        throw(
            ArgumentError(
                "Both or neither of `lowEnergyTail_A` and `lowEnergyTail_tau` must be provided",
            ),
        )
    xor(hasproperty(params, :highEnergyTail_A), hasproperty(params, :highEnergyTail_tau)) &&
        throw(
            ArgumentError(
                "Both or neither of `highEnergyTail_A` and `highEnergyTail_tau` must be provided",
            ),
        )

    needs_mu = has_peak || has_quadPoly || has_linPoly
    needs_mu && !hasproperty(params, :mu) && throw(ArgumentError("Need `mu` component"))

    needs_sigma = has_peak
    needs_sigma &&
        !hasproperty(params, :sigma) &&
        throw(ArgumentError("Need `sigma` component"))

    if has_peak
        gaussian_params =
            has_gaussian ?
            GaussianParams(A = params.gaussian_A, mu = params.mu, sigma = params.sigma) :
            false

        compton_params =
            has_compton ?
            ComptonParams(h = params.compton_h, mu = params.mu, sigma = params.sigma) :
            false

        lowEnergyTail_params =
            has_lowEnergyTail ?
            ExGaussianParams(
                A = params.lowEnergyTail_A,
                tau = params.lowEnergyTail_tau,
                is_lowEnergyTail = true,
                mu = params.mu,
                sigma = params.sigma,
            ) : false
        highEnergyTail_params =
            has_highEnergyTail ?
            ExGaussianParams(
                A = params.highEnergyTail_A,
                tau = params.highEnergyTail_tau,
                is_lowEnergyTail = false,
                mu = params.mu,
                sigma = params.sigma,
            ) : false

        peak = PeakParams(
            gaussian = gaussian_params,
            compton = compton_params,
            lowEnergyTail = lowEnergyTail_params,
            highEnergyTail = highEnergyTail_params,
        )
    else
        peak = nothing
    end

    if has_background
        quadPoly_params =
            has_quadPoly ? QuadPolyParams(C = params.quadPoly_C, mu = params.mu) : false

        linPoly_params =
            has_linPoly ? LinPolyParams(C = params.linPoly_C, mu = params.mu) : false

        constPoly_params = has_constPoly ? ConstPolyParams(C = params.constPoly_C) : false

        background = BackgroundParams(
            quadPoly = quadPoly_params,
            linPoly = linPoly_params,
            constPoly = constPoly_params,
        )
    else
        background = nothing
    end

    return ModelParams(peak = peak, background = background)
end

"""
    SpectrumData{T<:AbstractFloat, U<:Integer}

Container for binned energy spectrum data.

# Fields
- `bin_centers::AbstractVector{T}`: bin center(s) in keV
- `bin_edges::AbstractVector{T}`: bin edges in keV
- `weights::AbstractVector{U}`: observed count(s) per bin
- `bin_size::T`: width of each bin in keV

# Constructors

    SpectrumData(
        bin_markers::AbstractVector{T},
        weights::AbstractVector{U};
        bin_size::Union{T,Nothing} = nothing,
    ) where {T<:AbstractFloat, U<:Integer}

Construct a `SpectrumData` from bin markers and observed counts.

`bin_markers` are either the bin edges (`length(bin_markers) == length(weights) + 1`) or 
the bin centers (`length(bin_markers) == length(weights)`). The missing set is derived 
using `bin_size`. If `bin_size` is not supplied, it is estimated from the uniform bin 
markers.

# Arguments
- `bin_markers::AbstractVector{T}`: bin edges or bin centers in keV
- `weights::AbstractVector{U}`: observed counts per bin
- `bin_size::Union{T,Nothing}`: width of each bin in keV (optional). If `nothing`, the bin 
  size is estimated from `bin_markers`

# Throws
- An `ArgumentError` if `length(bin_markers)` is neither `length(weights)` nor
  `length(weights) + 1`
- An `ArgumentError` if `bin_size` is not supplied and only a single bin marker is given,
  since the bin size cannot be estimated
- An `ArgumentError` if `bin_size` is not supplied and the bins are not evenly spaced

---

    SpectrumData(
        lower_limit::T, 
        upper_limit::T, 
        bin_size::T, 
        params::ModelParams,
    ) where {T<:AbstractFloat}

Generate synthetic spectrum data from a model over a uniform grid of 
`(lower_limit):bin_size:(upper_limit)`, then sampling Poisson-distributed counts for each 
bin.

# Arguments
- `lower_limit::T`: start of the energy range in keV
- `upper_limit::T`: end of the energy range in keV
- `bin_size::T`: width of each bin in keV
- `params::ModelParams`: model parameters used to compute expected counts

# Throws
- An `ArgumentError` if the model produces negative expected counts

# Returns
- A `SpectrumData` object

# See also
- [`full_model`](@ref) for the used model
"""
Base.@kwdef struct SpectrumData{T<:AbstractFloat,U<:Integer}
    bin_centers::AbstractVector{T}
    bin_edges::AbstractVector{T}
    weights::AbstractVector{U}
    bin_size::T
end

function SpectrumData(
    bin_markers::AbstractVector{T},
    weights::AbstractVector{U};
    bin_size::Union{T,Nothing} = nothing,
) where {T<:AbstractFloat,U<:Integer}
    binMarker_length = length(bin_markers)
    weights_length = length(weights)

    if isnothing(bin_size)
        (binMarker_length == 1) && throw(
            ArgumentError(
                "Bin size can not be estimated, you must supply it directly via `bin_size`",
            ),
        )
        bin_sizes = bin_markers[2:end] - bin_markers[1:(end-1)]
        all(==(bin_sizes[1]), bin_sizes) ? bin_size = bin_sizes[1] :
        throw(ArgumentError("Your bins are not evenly spaced"))
    end

    if binMarker_length == weights_length
        bin_centers = bin_markers
        bin_edges =
            collect((bin_centers[1]-bin_size/2):bin_size:(bin_centers[end]+bin_size/2))
    elseif binMarker_length == (weights_length + 1)
        bin_edges = bin_markers
        bin_centers =
            collect((bin_edges[1]+bin_size/2):bin_size:(bin_edges[end]-bin_size/2))
    else
        throw(
            ArgumentError(
                "Your `bin_markers` and `weights` sizes do not match. Got $binMarker_length and $weights_length",
            ),
        )
    end

    return SpectrumData(
        bin_centers = bin_centers,
        bin_edges = bin_edges,
        weights = weights,
        bin_size = bin_size,
    )
end

function SpectrumData(
    lower_limit::T,
    upper_limit::T,
    bin_size::T,
    params::ModelParams,
) where {T<:AbstractFloat}
    bin_centers = collect(range(lower_limit, upper_limit; step = bin_size))
    bin_edges =
        collect(range(lower_limit - bin_size/2, upper_limit + bin_size/2; step = bin_size))
    expected_counts = full_model(bin_centers, params) .* bin_size
    any(expected_counts .< 0) &&
        throw(ArgumentError("Model produced negative expected counts; check parameters."))
    weights = rand.(Poisson.(expected_counts))
    return SpectrumData(
        bin_centers = bin_centers,
        bin_edges = bin_edges,
        weights = weights,
        bin_size = bin_size,
    )
end

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
    FitConfigs{T<:AbstractFloat}

Configuration options for the fitting process.

`mu` and `sigma` are required and have no default values. All remaining fields default to 
reasonable values.

# Fields
- `mu::T`: expected centroid position of the peak in keV
- `sigma::T`: expected standard deviation of the Gaussian core in keV
- `integration_method::Symbol`: the integration method used to compute expected bin counts.
  Either `:analytical`, `:numerical`, or `:midpoint`. Default: `:analytical`

# See also
- [`PriorConfigs`](@ref) for the configuration options for the priors
"""
Base.@kwdef struct FitConfigs{T<:AbstractFloat}
    mu::T
    sigma::T
    integration_method::Symbol = :analytical
    prior::PriorConfigs = PriorConfigs()
end
