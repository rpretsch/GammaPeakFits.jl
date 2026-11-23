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
    _peak_mask(bin_centers::AbstractVector{<:Real}; mu::Real, sigma::Real) -> BitVector

Select the peak region of a spectrum: the bins whose centers lie within `mu ± 3 sigma`.
The region must lie within the range of the bin centers, contain at least one bin center,
and leave at least one bin outside for the background estimate.

# Arguments
- `bin_centers::AbstractVector{<:Real}`: bin centers in keV

# Keyword Arguments
- `mu::Real`: expected peak centroid in keV
- `sigma::Real`: expected peak width (standard deviation) in keV

# Throws
- An `ArgumentError` if `sigma` is not positive
- An `ArgumentError` if `mu ± 3 sigma` is not within the range of the bin centers
- An `ArgumentError` if the peak region contains no bin center
- An `ArgumentError` if the peak region contains all bin centers

# Returns
- A `BitVector` that is `true` for the bins in the peak region
"""
function _peak_mask(bin_centers::AbstractVector{<:Real}; mu::Real, sigma::Real)

    sigma > 0 || throw(ArgumentError("`sigma` must be positive, got $sigma"))

    lower = mu - 3 * sigma
    upper = mu + 3 * sigma
    data_min, data_max = extrema(bin_centers)
    data_min <= lower && upper <= data_max || throw(
        ArgumentError(
            "Peak region [$lower, $upper] keV is not fully contained in the data range " *
            "[$data_min, $data_max] keV",
        ),
    )

    mask = lower .<= bin_centers .<= upper
    any(mask) || throw(
        ArgumentError(
            "Peak region [$lower, $upper] keV contains no bin centers, use a larger " *
            "`sigma` or finer bins",
        ),
    )
    all(mask) && throw(
        ArgumentError(
            "Data range [$data_min, $data_max] keV contains only the peak region " *
            "[$lower, $upper] keV, need bins outside for the background",
        ),
    )

    return mask

end

"""
    _get_peak_features(hist::Histogram{<:Real,1}; mu::Real, sigma::Real) -> Tuple{Real,Real,Real}

Estimate the peak height and area from observed count data.

The peak region is selected by [`_peak_mask`](@ref). First, the mean background is
calculated from the bins outside peak region. Then the height is taken as the maximum
observed count in the peak area minus the mean background. The peak area is estimated as
`sqrt(2 * pi) * sigma * peak_height`.

Poisson data can produce zero background counts or no peak above the background, so 
`peak_height` and `mean_background` are floored at one count per bin; `peak_area` is 
derived from the floored height. This keeps the `Uniform` bounds of [`peak_priors`](@ref)
and [`background_priors`](@ref) valid.

# Arguments
- `hist::Histogram{<:Real,1}`: binned spectrum, see [`_check_spectrum`](@ref)

# Keyword Arguments
- `mu::Real`: expected peak centroid in keV
- `sigma::Real`: expected peak width (standard deviation) in keV

# Returns
- A tuple `(peak_height, peak_area, mean_background)` containing the estimated height and 
  area of the peak, as well as the estimated mean background in 
  (counts/keV, counts, counts/keV)

# Throws
- The `ArgumentError`s of [`_check_spectrum`](@ref) and [`_peak_mask`](@ref)
"""
function _get_peak_features(hist::Histogram{<:Real,1}; mu::Real, sigma::Real)

    bin_size = _check_spectrum(hist)
    weights = hist.weights
    peak_mask = _peak_mask(midpoints(only(hist.edges)); mu, sigma)
    peak_weights = weights[peak_mask]                                   # counts/bin

    background_weights = weights[.!peak_mask]                           # counts/bin
    mean_background_val = _mean_background(background_weights, bin_size) # counts/keV
    peak_height =
        max(maximum(peak_weights) / bin_size - mean_background_val, 1.0 / bin_size)
    # counts/keV
    peak_area = sqrt(2 * pi) * sigma * peak_height                      # counts

    return peak_height, peak_area, mean_background_val

end
