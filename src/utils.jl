"""
    cut_data(hist::Histogram{<:Real,1}; center::Real, width::Real) -> Histogram

Slice a spectrum to a fit window, e.g. around a peak. Only bins that lie entirely within
`[center - width/2, center + width/2]` are retained.

# Arguments
- `hist::Histogram{<:Real,1}`: binned spectrum, see [`_check_spectrum`](@ref)

# Keyword Arguments
- `center::Real`: center of the fit window in keV, e.g. the expected peak centroid
- `width::Real`: full width of the fit window in keV

# Throws
- The `ArgumentError`s of [`_check_spectrum`](@ref)
- An `ArgumentError` if the window contains no complete bin

# Returns
- A new `Histogram` containing only the bins in the window
"""
function cut_data(hist::Histogram{<:Real,1}; center::Real, width::Real)

    _check_spectrum(hist)

    edges = only(hist.edges)
    lower = center - width / 2
    upper = center + width / 2
    mask_edges = lower .<= edges .<= upper
    mask_bins = mask_edges[1:(end-1)] .& mask_edges[2:end]

    any(mask_bins) ||
        throw(ArgumentError("The fit window [$lower, $upper] keV contains no complete bin"))

    return Histogram(edges[mask_edges], hist.weights[mask_bins], hist.closed)

end

# Mean background level in counts/keV from per-bin counts, floored at one count per bin so
# that the `Uniform` bounds of the default priors stay valid:
function _mean_background(weights::AbstractVector{<:Real}, bin_size::Real)

    return max(mean(weights), 1.0) / bin_size

end

"""
    _background_poly_bounds(mean_background::Real, half_window::Real) -> Tuple{Real,Real}

Compute symmetric prior bounds for the linear and quadratic background coefficients.

The polynomial background is expanded around the center `x_0` of the fit window,

```math
f_{\\text{bg}}(x) = C_q\\,(x - x_0)^2 + C_l\\,(x - x_0) + C_c,
```

so the coefficients `C_l` (counts/keV²) and `C_q` (counts/keV³) are dimensionful. Their
natural scale is set by the mean background level `mean_background` (counts/keV) and the
half-width of the fit window `h = half_window` (keV): the bounds are chosen so
that each term can vary by up to twice the background level across the window, giving

```math
|C_l| \\le \\frac{2\\,\\text{mean\\_background}}{h}, \\qquad
|C_q| \\le \\frac{2\\,\\text{mean\\_background}}{h^2}.
```

# Arguments
- `mean_background::Real`: estimated mean background level in counts/keV
- `half_window::Real`: half-width of the fit window in keV

# Returns
- A `Tuple` `(lin_bound, quad_bound)` with the positive bounds for `C_l` and `C_q`
"""
function _background_poly_bounds(mean_background::Real, half_window::Real)

    lin_bound = 2 * mean_background / half_window
    quad_bound = 2 * mean_background / half_window^2

    return lin_bound, quad_bound

end

"""
    _peak_params(
        mu::Union{Real,AbstractVector{<:Real}},
        sigma::Union{Real,AbstractVector{<:Real}},
    ) -> Tuple{Vector,Vector}

The known peaks in a spectrum as two vectors of equal length, their expected centroids and
widths (standard deviations) in keV. `mu` is one centroid or a vector of centroids; `sigma`
is one width shared by all peaks or a vector with one width per peak. All known peaks in
the window are given, also those that are not fitted, so that they are excluded from the
background estimate (see [`_background_mask`](@ref)).

# Arguments
- `mu::Union{Real,AbstractVector{<:Real}}`: expected peak centroid(s) in keV
- `sigma::Union{Real,AbstractVector{<:Real}}`: expected peak width(s) in keV

# Throws
- An `ArgumentError` if `mu` is empty
- An `ArgumentError` if `sigma` is a vector of another length than `mu`
- An `ArgumentError` if a `sigma` is not positive

# Returns
- A `Tuple` `(mus, sigmas)` of two vectors with one entry per peak
"""
function _peak_params(
    mu::Union{Real,AbstractVector{<:Real}},
    sigma::Union{Real,AbstractVector{<:Real}},
)

    mus = mu isa Real ? [float(mu)] : float.(mu)
    sigmas = sigma isa Real ? fill(float(sigma), length(mus)) : float.(sigma)

    isempty(mus) && throw(ArgumentError("`mu` must contain at least one peak"))
    length(sigmas) == length(mus) || throw(
        ArgumentError(
            "`sigma` must be one width or one width per peak, got $(length(sigmas)) " *
            "widths for $(length(mus)) peaks",
        ),
    )
    all(>(0), sigmas) || throw(ArgumentError("`sigma` must be positive, got $sigma"))

    return mus, sigmas

end

"""
    _peak_mask(bin_centers::AbstractVector{<:Real}; mu::Real, sigma::Real) -> BitVector

Select the peak region of one peak: the bins whose centers lie within `mu ± 3 sigma`. The
region must contain at least one bin center.

# Arguments
- `bin_centers::AbstractVector{<:Real}`: bin centers in keV

# Keyword Arguments
- `mu::Real`: expected peak centroid in keV
- `sigma::Real`: expected peak width (standard deviation) in keV

# Throws
- An `ArgumentError` if the peak region contains no bin center

# Returns
- A `BitVector` that is `true` for the bins in the peak region
"""
function _peak_mask(bin_centers::AbstractVector{<:Real}; mu::Real, sigma::Real)

    lower = mu - 3 * sigma
    upper = mu + 3 * sigma
    mask = lower .<= bin_centers .<= upper
    any(mask) || throw(
        ArgumentError(
            "Peak region [$lower, $upper] keV contains no bin centers, use a larger " *
            "`sigma` or finer bins",
        ),
    )

    return mask

end

"""
    _background_mask(
        bin_centers::AbstractVector{<:Real};
        mu::AbstractVector{<:Real},
        sigma::AbstractVector{<:Real},
    ) -> BitVector

Select the background region of a spectrum: the bins whose centers lie outside
`mu ± 5 sigma` of every known peak (see [`_peak_params`](@ref)). Bins between `3 sigma`
and `5 sigma` belong to neither the peak region ([`_peak_mask`](@ref)) nor the background,
so that the peak tails do not raise the background estimate. At least one bin must remain.

# Arguments
- `bin_centers::AbstractVector{<:Real}`: bin centers in keV

# Keyword Arguments
- `mu::AbstractVector{<:Real}`: expected centroids of all known peaks in keV
- `sigma::AbstractVector{<:Real}`: expected widths of all known peaks in keV, one per peak

# Throws
- An `ArgumentError` if no bin center lies outside the excluded regions

# Returns
- A `BitVector` that is `true` for the bins in the background region
"""
function _background_mask(
    bin_centers::AbstractVector{<:Real};
    mu::AbstractVector{<:Real},
    sigma::AbstractVector{<:Real},
)

    mask = trues(length(bin_centers))
    for (m, s) in zip(mu, sigma)
        mask .&= abs.(bin_centers .- m) .> 5 * s
    end
    any(mask) || throw(
        ArgumentError(
            "No bin lies outside `mu ± 5 sigma` of the peaks, use a wider window for " *
            "the background",
        ),
    )

    return mask

end

"""
    _get_peak_features(
        hist::Histogram{<:Real,1};
        mu::Union{Real,AbstractVector{<:Real}},
        sigma::Union{Real,AbstractVector{<:Real}},
        peak_index::Integer = 1,
    ) -> Tuple{Real,Real,Real}

Estimate the height and area of one peak and the mean background from observed count data.

`mu` and `sigma` describe all known peaks ([`_peak_params`](@ref)), `peak_index` selects
the one to estimate. The mean background is taken over the background region of all peaks
([`_background_mask`](@ref)); the height is the maximum count in the peak region of the
selected peak ([`_peak_mask`](@ref)) minus the mean background, the area is
`sqrt(2 * pi) * sigma * peak_height` with the sigma of the selected peak. A second peak
within the peak region of the selected one is not separated: the maximum may then come from
it.

`peak_height` and `mean_background` are floored at one count per bin, because Poisson data
can lack background counts or a peak above the background; this keeps the `Uniform` bounds
of [`peak_priors`](@ref) and [`background_priors`](@ref) valid.

# Arguments
- `hist::Histogram{<:Real,1}`: binned spectrum, see [`_check_spectrum`](@ref)

# Keyword Arguments
- `mu::Union{Real,AbstractVector{<:Real}}`: expected centroid(s) of the known peaks in keV,
  see [`_peak_params`](@ref)
- `sigma::Union{Real,AbstractVector{<:Real}}`: expected width(s) of the known peaks in keV,
  see [`_peak_params`](@ref)
- `peak_index::Integer = 1`: index of the peak to estimate in `mu`

# Returns
- A tuple `(peak_height, peak_area, mean_background)` containing the estimated height and
  area of the selected peak, as well as the estimated mean background in
  (counts/keV, counts, counts/keV)

# Throws
- The `ArgumentError`s of [`_check_spectrum`](@ref), [`_peak_params`](@ref),
  [`_peak_mask`](@ref) and [`_background_mask`](@ref)
- An `ArgumentError` if `peak_index` is not an index of `mu`
"""
function _get_peak_features(
    hist::Histogram{<:Real,1};
    mu::Union{Real,AbstractVector{<:Real}},
    sigma::Union{Real,AbstractVector{<:Real}},
    peak_index::Integer = 1,
)

    bin_size = _check_spectrum(hist)
    mus, sigmas = _peak_params(mu, sigma)
    peak_index in eachindex(mus) || throw(
        ArgumentError(
            "`peak_index` must be an index of `mu` ($(eachindex(mus))), got $peak_index",
        ),
    )
    peak_mu, peak_sigma = mus[peak_index], sigmas[peak_index]

    weights = hist.weights
    bin_centers = midpoints(only(hist.edges))
    peak_mask = _peak_mask(bin_centers; mu = peak_mu, sigma = peak_sigma)
    background_mask = _background_mask(bin_centers; mu = mus, sigma = sigmas)

    peak_weights = weights[peak_mask]                                       # counts/bin
    background_weights = weights[background_mask]                           # counts/bin
    mean_background_val = _mean_background(background_weights, bin_size)    # counts/keV
    peak_height =
        max(maximum(peak_weights) / bin_size - mean_background_val, 1.0 / bin_size)
    # counts/keV
    peak_area = sqrt(2 * pi) * peak_sigma * peak_height                     # counts

    return peak_height, peak_area, mean_background_val

end
