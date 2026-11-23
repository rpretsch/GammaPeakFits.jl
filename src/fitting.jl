"""
    build_prior(
        hist::Histogram{<:Real,1},
        model::PropertyFunction,
        configs::FitConfigs
    ) -> NamedTupleDist

Construct a prior over the parameters of a `@pf` model kernel, e.g.
`@pf \$n * GaussPeak(\$mu, \$sigma) + PolynomialShape(\$b)`.

The parameter names are read from the kernel. Known names get a weakly informative,
data-driven default (see Details), `configs.priors` overrides any of them. The prior
follows the order in which the names first appear in the kernel.

Peak features are estimated only if the kernel has a peak amplitude `n`; otherwise the
background level is estimated over the whole data range, so no peak region needs to be
contained in the data.

# Arguments
- `hist::Histogram{<:Real,1}`: binned spectrum, see [`_check_spectrum`](@ref)
- `model::PropertyFunction`: model kernel `params -> measure`, built with `@pf`
- `configs::FitConfigs`: fitting configuration

# Returns
- A `NamedTupleDist` over the parameters of `model`

# Throws
- The `ArgumentError`s of [`_check_spectrum`](@ref) and, with `n`, of
  [`get_peak_features`](@ref)
- An `ArgumentError` if a parameter is nested (`\$a.b`), which is not supported yet
- An `ArgumentError` if a parameter has neither a default nor an entry in `configs.priors`
- An `ArgumentError` if `configs.priors` has an entry for a name not in the kernel

# Details

| Name    | Default prior                                       |
| ------- | --------------------------------------------------- |
| `n`     | `Uniform(0, 3 * peak_area)`                         |
| `alpha` | `Uniform(0, 1)`                                     |
| `mu`    | `Normal(configs.mu, 0.6)`                           |
| `sigma` | `truncated(Normal(configs.sigma, 0.6), eps(), Inf)` |
| `theta` | `Uniform(eps(), 5 * configs.sigma)`                 |
| `b`     | `Uniform(0, 2 * mean_background)`                   |
| `b1`    | `Uniform(-lin_bound, lin_bound)`                    |
| `b2`    | `Uniform(-quad_bound, quad_bound)`                  |
| `h`     | `Uniform(0, 4 * mean_background)`                   |

`peak_area` and `mean_background` come from [`get_peak_features`](@ref) /
[`mean_background`](@ref), `lin_bound` and `quad_bound` from
[`background_poly_bounds`](@ref). These bounds assume that the kernel expands the
polynomial background around the window center, e.g.
`PolynomialShape(\$b, \$b1, \$b2; x0 = configs.mu)`.
"""
function build_prior(
    hist::Histogram{<:Real,1},
    model::PropertyFunction,
    configs::FitConfigs,
)

    bin_size = _check_spectrum(hist)
    names = map(_parameter_name, input_property_paths(model))
    overrides = Dict(configs.priors)

    unused = setdiff(keys(overrides), names)
    isempty(unused) || throw(
        ArgumentError(
            "`configs.priors` has entries for names not in the model: $(collect(unused))",
        ),
    )

    if :n in names
        _, peak_area, background =
            get_peak_features(hist; mu = configs.mu, sigma = configs.sigma)
    else
        peak_area = NaN  # not used without a peak amplitude `n`
        background = mean_background(hist.weights, bin_size)
    end
    lin_bound, quad_bound =
        background_poly_bounds(background, max(configs.window_size / 2, bin_size))

    priors = map(names) do name
        haskey(overrides, name) ? overrides[name] :
        _default_prior(name, configs, peak_area, background, lin_bound, quad_bound)
    end

    return distprod(; NamedTuple{names}(priors)...)

end

# Flat parameter name of a kernel input path, `$n` -> `:n`:
function _parameter_name(::PPath{path}) where {path}
    length(path) == 1 || throw(
        ArgumentError("Nested model parameter `$(join(path, '.'))` is not supported yet"),
    )
    return only(path)
end

# Default prior of a known parameter name, see `build_prior`:
function _default_prior(name::Symbol, configs, peak_area, background, lin_bound, quad_bound)
    name === :n && return Uniform(0, 3 * peak_area)
    name === :alpha && return Uniform(0, 1)
    name === :mu && return Normal(configs.mu, 0.6)
    name === :sigma && return truncated(Normal(configs.sigma, 0.6), eps(), Inf)
    name === :theta && return Uniform(eps(), 5 * configs.sigma)
    name === :b && return Uniform(0, 2 * background)
    name === :b1 && return Uniform(-lin_bound, lin_bound)
    name === :b2 && return Uniform(-quad_bound, quad_bound)
    name === :h && return Uniform(0, 4 * background)
    throw(ArgumentError("No default prior for `$name`, supply one via `configs.priors`"))
end

"""
    build_posterior(
        hist::Histogram{<:Real,1},
        model::PropertyFunction,
        prior::NamedTupleDist
    ) -> PosteriorMeasure

Construct the posterior of a `@pf` model kernel for an observed spectrum: the binned
Poisson likelihood of `hist` (LegendSpecFits' `binned_poisson_likelihood`) and `prior`.

# Arguments
- `hist::Histogram{<:Real,1}`: observed spectrum, see [`_check_spectrum`](@ref)
- `model::PropertyFunction`: model kernel `params -> measure`, built with `@pf`
- `prior::NamedTupleDist`: prior over the parameters of `model`, e.g. from
  [`build_prior`](@ref)

# Throws
- The `ArgumentError`s of [`_check_spectrum`](@ref)
- An `ArgumentError` if a parameter of `model` is nested (`\$a.b`), which is not
  supported yet
- An `ArgumentError` if the names of `prior` differ from the parameters of `model`

# Returns
- A `PosteriorMeasure` of the likelihood and `prior`
"""
function build_posterior(
    hist::Histogram{<:Real,1},
    model::PropertyFunction,
    prior::NamedTupleDist,
)

    _check_spectrum(hist)

    names = map(_parameter_name, input_property_paths(model))
    issetequal(names, propertynames(prior)) || throw(
        ArgumentError(
            "The prior's parameters $(propertynames(prior)) differ from the model's $names",
        ),
    )

    return PosteriorMeasure(binned_poisson_likelihood(model, hist), prior)

end
