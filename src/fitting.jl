"""
    peak_priors(hist::Histogram{<:Real,1}; mu::Real, sigma::Real) -> NamedTuple

Data-driven default priors for the parameters of one peak, estimated from `hist` around
the expected centroid `mu` and width `sigma`:

| Name    | Default prior                               |
| ------- | ------------------------------------------- |
| `n`     | `Uniform(0, 3 * peak_area)`                 |
| `alpha` | `Uniform(0, 1)`                             |
| `mu`    | `Normal(mu, 0.6)`                           |
| `sigma` | `truncated(Normal(sigma, 0.6), eps(), Inf)` |
| `theta` | `Uniform(eps(), 5 * sigma)`                 |

`peak_area` is estimated by [`_get_peak_features`](@ref). Entries a model does not use (e.g. `alpha`, `theta` without a tail) are ignored by
[`build_prior`](@ref); single entries are replaced with `merge`.

# Arguments
- `hist::Histogram{<:Real,1}`: binned spectrum, see [`_check_spectrum`](@ref), e.g. from
  [`cut_data`](@ref)

# Keyword Arguments
- `mu::Real`: expected peak centroid in keV
- `sigma::Real`: expected peak width (standard deviation) in keV

# Throws
- The `ArgumentError`s of [`_get_peak_features`](@ref)

# Returns
- A `NamedTuple` `(n, alpha, mu, sigma, theta)` of distributions
"""
function peak_priors(hist::Histogram{<:Real,1}; mu::Real, sigma::Real)

    _, peak_area, _ = _get_peak_features(hist; mu, sigma)

    return (
        n = Uniform(0, 3 * peak_area),
        alpha = Uniform(0, 1),
        mu = Normal(mu, 0.6),
        sigma = truncated(Normal(sigma, 0.6), eps(), Inf),
        theta = Uniform(eps(), 5 * sigma),
    )

end

"""
    background_priors(
        hist::Histogram{<:Real,1};
        mu::Union{Real,Nothing} = nothing,
        sigma::Union{Real,Nothing} = nothing,
    ) -> NamedTuple

Data-driven default priors for the background parameters, from the mean background level
`level` of `hist` (counts/keV) and the half-width `h` of its energy range:

| Name | Default prior                    | Meaning                                  |
| ---- | -------------------------------- | ---------------------------------------- |
| `b`  | `Uniform(0, 2 * level)`          | constant term                            |
| `b1` | `Uniform(-2 * level / h, ...)`   | linear term around the window center     |
| `b2` | `Uniform(-2 * level / h^2, ...)` | quadratic term around the window center  |
| `h`  | `Uniform(0, 4 * level)`          | Compton step height                      |

With `mu` and `sigma`, the level is estimated outside the peak region by
[`_get_peak_features`](@ref), otherwise over all bins, floored at one count per bin. The
`b1`/`b2` bounds are those of [`_background_poly_bounds`](@ref); they assume that the
model expands the polynomial around the center of the range, e.g.
`PolynomialShape(\$b, \$b1, \$b2; x0 = center)`.

# Arguments
- `hist::Histogram{<:Real,1}`: binned spectrum, see [`_check_spectrum`](@ref), e.g. from
  [`cut_data`](@ref)

# Keyword Arguments
- `mu::Union{Real,Nothing} = nothing`: centroid of a peak to exclude, in keV
- `sigma::Union{Real,Nothing} = nothing`: width of a peak to exclude, in keV

# Throws
- An `ArgumentError` if only one of `mu` and `sigma` is given
- The `ArgumentError`s of [`_check_spectrum`](@ref)
- With `mu` and `sigma`: the `ArgumentError`s of [`_get_peak_features`](@ref)

# Returns
- A `NamedTuple` `(b, b1, b2, h)` of distributions
"""
function background_priors(
    hist::Histogram{<:Real,1};
    mu::Union{Real,Nothing} = nothing,
    sigma::Union{Real,Nothing} = nothing,
)

    isnothing(mu) == isnothing(sigma) || throw(
        ArgumentError("Give both `mu` and `sigma` to exclude a peak region, or neither"),
    )

    bin_size = _check_spectrum(hist)
    level = if isnothing(mu)
        _mean_background(hist.weights, bin_size)
    else
        last(_get_peak_features(hist; mu, sigma))
    end

    edges = only(hist.edges)
    lin_bound, quad_bound = _background_poly_bounds(level, (last(edges) - first(edges)) / 2)

    return (
        b = Uniform(0, 2 * level),
        b1 = Uniform(-lin_bound, lin_bound),
        b2 = Uniform(-quad_bound, quad_bound),
        h = Uniform(0, 4 * level),
    )

end

"""
    build_prior(model::PropertyFunction, defaults::NamedTuple) -> NamedTupleDist

Construct the prior over the parameters of a `@pf` model kernel from a, possibly nested,
`NamedTuple` of default priors, e.g. from [`peak_priors`](@ref) and
[`background_priors`](@ref).

Each parameter path of `model` takes the matching entry of `defaults`: `\$n` takes
`defaults.n`, `\$peak.n` takes `defaults.peak.n`; groups become nested priors. The prior
follows the order in which the names first appear in the kernel. Entries the model does
not use are ignored; single entries are replaced with `merge` beforehand.

# Example

    model = @pf \$peak.n * GaussPeak(\$peak.mu, \$peak.sigma) + PolynomialShape(\$bkg.b)
    defaults = (
        peak = merge(peak_priors(hist; mu = 2614.5, sigma = 1.2), (n = Uniform(0, 1e5),)),
        bkg = background_priors(hist; mu = 2614.5, sigma = 1.2),
    )
    prior = build_prior(model, defaults)

# Arguments
- `model::PropertyFunction`: model kernel `params -> measure`, built with `@pf`
- `defaults::NamedTuple`: distributions per parameter name, `NamedTuple`s per group

# Throws
- An `ArgumentError` if a parameter path of `model` has no entry in `defaults`
- An `ArgumentError` if the entry of a parameter group is not a `NamedTuple`

# Returns
- A `NamedTupleDist` over the parameters of `model`, nested like its parameter groups
"""
function build_prior(model::PropertyFunction, defaults::NamedTuple)
    paths = map(_property_path, input_property_paths(model))
    return _select_priors(paths, defaults)
end

# Property path of a kernel input as a tuple of names, `$peak.n` -> `(:peak, :n)`:
_property_path(::PPath{path}) where {path} = path

# Prior for the kernel `paths` from `defaults`; `group` is the path of the current group
# in `defaults`, only for the error messages of the recursion:
function _select_priors(paths, defaults::NamedTuple; group::Tuple = ())
    names = unique(map(first, paths))
    priors = map(names) do name
        path = (group..., name)
        haskey(defaults, name) ||
            throw(ArgumentError("No prior for `$(join(path, '.'))` in `defaults`"))
        entry = defaults[name]
        rests = [Base.tail(p) for p in paths if first(p) === name]
        any(isempty, rests) && return entry  # the kernel uses the entry as a whole
        entry isa NamedTuple || throw(
            ArgumentError(
                "The entry for the group `$(join(path, '.'))` must be a `NamedTuple`, got $(typeof(entry))",
            ),
        )
        return _select_priors(rests, entry; group = path)
    end
    return distprod(; NamedTuple{Tuple(names)}(priors)...)
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
- An `ArgumentError` if the names of `prior` differ from the parameters of `model`, on any
  level of its parameter groups

# Returns
- A `PosteriorMeasure` of the likelihood and `prior`
"""
function build_posterior(
    hist::Histogram{<:Real,1},
    model::PropertyFunction,
    prior::NamedTupleDist,
)

    _check_spectrum(hist)
    _check_prior(prior, map(_property_path, input_property_paths(model)))

    return PosteriorMeasure(binned_poisson_likelihood(model, hist), prior)

end

# Check that `prior` has exactly the parameters of the kernel `paths`, on every level;
# `group` is the path of the current group, only for the error messages of the recursion:
function _check_prior(prior::NamedTupleDist, paths; group::Tuple = ())
    names = Tuple(unique(map(first, paths)))
    level = isempty(group) ? "" : " in `$(join(group, '.'))`"
    issetequal(names, propertynames(prior)) || throw(
        ArgumentError(
            "The prior's parameters $(propertynames(prior))$level differ from the model's $names",
        ),
    )
    for name in names
        rests = [Base.tail(p) for p in paths if first(p) === name]
        any(isempty, rests) && continue
        group_prior = getproperty(prior, name)
        group_prior isa NamedTupleDist || throw(
            ArgumentError(
                "The prior of the group `$(join((group..., name), '.'))` must be nested",
            ),
        )
        _check_prior(group_prior, rests; group = (group..., name))
    end
end
