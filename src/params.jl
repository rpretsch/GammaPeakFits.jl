"""
    AbstractComponent

Abstract supertype for everything that can occupy a model-component slot: the presence 
markers [`Enabled`](@ref)/[`Disabled`](@ref), the concrete parameter structs
([`GaussianParams`](@ref), ...), and the container types [`PeakParams`](@ref)/
[`BackgroundParams`](@ref).

# See also
- [`is_present`](@ref), [`Enabled`](@ref), [`Disabled`](@ref)
"""
abstract type AbstractComponent end

"""
    Enabled

Marker to include a model component in the fit without setting its parameters.

# See also
- [`Disabled`](@ref), [`AbstractComponent`](@ref)
"""
struct Enabled <: AbstractComponent end

"""
    Disabled

Marker that a model component (or an entire peak/background block) is excluded from the 
model.

When present in a slot, the component contributes zero to the model and its integrals.

# See also
- [`Enabled`](@ref), [`AbstractComponent`](@ref)
"""
struct Disabled <: AbstractComponent end

"""
    GaussianParams{T<:AbstractFloat} <: AbstractComponent

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
Base.@kwdef struct GaussianParams{T<:AbstractFloat} <: AbstractComponent
    A::T
    mu::T
    sigma::T
end

"""
    ComptonParams{T<:AbstractFloat} <: AbstractComponent

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
Base.@kwdef struct ComptonParams{T<:AbstractFloat} <: AbstractComponent
    h::T
    mu::T
    sigma::T
end

"""
    ExGaussianParams{T<:AbstractFloat} <: AbstractComponent

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
Base.@kwdef struct ExGaussianParams{T<:AbstractFloat} <: AbstractComponent
    A::T
    tau::T
    is_lowEnergyTail::Bool
    mu::T
    sigma::T
end

"""
    PeakParams{
        G<:AbstractComponent, 
        C<:AbstractComponent,
        L<:AbstractComponent, 
        H<:AbstractComponent
    } <: AbstractComponent

Aggregate container for all components that form a gamma-ray peak.

Each component is optional — set the corresponding field to `Disabled()` to exclude it. 
Setting it to `Enabled()` instead of specifying a `XParams` object allows for controlling 
which component is used in the fitting process (See [build_prior](@ref)).

`mu` and `sigma` are usually the same between all model components.

# Fields
- `gaussian::G`: main Gaussian peak shape. Default: `Disabled()`
- `compton::C`: Compton-edge step function. Default: `Disabled()`
- `lowEnergyTail::L`: ex-Gaussian low-energy tail. Default: `Disabled()`
- `highEnergyTail::H`: ex-Gaussian high-energy tail. Default: `Disabled()`

# See also
- [`Enabled`](@ref), [`Disabled`](@ref), [`AbstractComponent`](@ref) for the component 
  management
- [`peak_model`](@ref) for evaluating the combined peak shape
- [`GaussianParams`](@ref), [`ComptonParams`](@ref), and [`ExGaussianParams`](@ref) for the
  component parameters.
"""
Base.@kwdef struct PeakParams{
    G<:AbstractComponent,
    C<:AbstractComponent,
    L<:AbstractComponent,
    H<:AbstractComponent,
} <: AbstractComponent
    gaussian::G = Disabled()
    compton::C = Disabled()
    lowEnergyTail::L = Disabled()
    highEnergyTail::H = Disabled()
end

"""
    QuadPolyParams{T<:AbstractFloat} <: AbstractComponent

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
Base.@kwdef struct QuadPolyParams{T<:AbstractFloat} <: AbstractComponent
    C::T
    mu::T
end

"""
    LinPolyParams{T<:AbstractFloat} <: AbstractComponent

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
Base.@kwdef struct LinPolyParams{T<:AbstractFloat} <: AbstractComponent
    C::T
    mu::T
end

"""
    ConstPolyParams{T<:AbstractFloat} <: AbstractComponent

Parameters for the constant polynomial term in the background model.

# Fields
- `C::T`: constant offset in counts/keV

# See also
- [`const_polynomial`](@ref) for evaluating the term
"""
Base.@kwdef struct ConstPolyParams{T<:AbstractFloat} <: AbstractComponent
    C::T
end

"""
    BackgroundParams{
        Q<:AbstractComponent, 
        L<:AbstractComponent, 
        C<:AbstractComponent
    } <: AbstractComponent

Aggregate container for all components that form the background model.

Each component is optional — set the corresponding field to `Disabled()` to exclude it. 
Setting it to `Enabled()` instead of specifying a `XParams` object allows for controlling 
which component is used in the fitting process (See [build_prior](@ref)).

`mu` and `sigma` are usually the same between all model components.

# Fields
- `quadPoly::Q`: quadratic polynomial term. Default: `Disabled()`
- `linPoly::C`: linear polynomial term. Default: `Disabled()`
- `constPoly::L`: constant polynomial term. Default: `Disabled()`

# See also
- [`Enabled`](@ref), [`Disabled`](@ref), [`AbstractComponent`](@ref) for the component 
  management
- [background_model](@ref) for evaluating the background model
- [`QuadPolyParams`](@ref), [`ComptonParams`](@ref), and [`ConstPolyParams`](@ref) for the
  component parameters.
"""
Base.@kwdef struct BackgroundParams{
    Q<:AbstractComponent,
    L<:AbstractComponent,
    C<:AbstractComponent,
} <: AbstractComponent
    quadPoly::Q = Disabled()
    linPoly::L = Disabled()
    constPoly::C = Disabled()
end

"""
    ModelParams{P<:AbstractComponent, B<:AbstractComponent}

Complete model combining a gamma peak and a polynomial background.

Each component is optional — unset fields are `Disabled()` and are skipped during 
evaluation.

`mu` and `sigma` are usually the same between all model components.

# Fields
- `peak::P`: peak term. Default: `Disabled()`
- `background::B`: background term. Default: `Disabled()`

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
- [`Enabled`](@ref), [`Disabled`](@ref), [`AbstractComponent`](@ref) for the component 
  management
- [`full_model`](@ref) for evaluating the combined model
- [`PeakParams`](@ref), and [`BackgroundParams`](@ref) for the component parameters
- [`build_prior`](@ref) for constructing the `NamedTuple` distribution this constructor
  consumes
"""
Base.@kwdef struct ModelParams{P<:AbstractComponent,B<:AbstractComponent}
    peak::P = Disabled()
    background::B = Disabled()
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
            Disabled()

        compton_params =
            has_compton ?
            ComptonParams(h = params.compton_h, mu = params.mu, sigma = params.sigma) :
            Disabled()

        lowEnergyTail_params =
            has_lowEnergyTail ?
            ExGaussianParams(
                A = params.lowEnergyTail_A,
                tau = params.lowEnergyTail_tau,
                is_lowEnergyTail = true,
                mu = params.mu,
                sigma = params.sigma,
            ) : Disabled()
        highEnergyTail_params =
            has_highEnergyTail ?
            ExGaussianParams(
                A = params.highEnergyTail_A,
                tau = params.highEnergyTail_tau,
                is_lowEnergyTail = false,
                mu = params.mu,
                sigma = params.sigma,
            ) : Disabled()

        peak = PeakParams(
            gaussian = gaussian_params,
            compton = compton_params,
            lowEnergyTail = lowEnergyTail_params,
            highEnergyTail = highEnergyTail_params,
        )
    else
        peak = Disabled()
    end

    if has_background
        quadPoly_params =
            has_quadPoly ? QuadPolyParams(C = params.quadPoly_C, mu = params.mu) :
            Disabled()

        linPoly_params =
            has_linPoly ? LinPolyParams(C = params.linPoly_C, mu = params.mu) : Disabled()

        constPoly_params =
            has_constPoly ? ConstPolyParams(C = params.constPoly_C) : Disabled()

        background = BackgroundParams(
            quadPoly = quadPoly_params,
            linPoly = linPoly_params,
            constPoly = constPoly_params,
        )
    else
        background = Disabled()
    end

    return ModelParams(peak = peak, background = background)
end
