"""
    cut_data(hist::Histogram{<:Real,1}, configs::FitConfigs) -> Histogram

Slice a spectrum to the fit window centered on a peak.

Only bins that lie entirely within
`[configs.mu - configs.window_size/2, configs.mu + configs.window_size/2]` are retained.

# Arguments
- `hist::Histogram{<:Real,1}`: binned spectrum, see [`_check_spectrum`](@ref)
- `configs::FitConfigs`: fitting configuration providing the centroid `mu` and the full
  window width `window_size`

# Throws
- The `ArgumentError`s of [`_check_spectrum`](@ref)
- An `ArgumentError` if the window contains no complete bin

# Returns
- A new `Histogram` containing only the bins in the selected window

# See also
- [`FitConfigs`](@ref) for the configuration options
"""
function cut_data(hist::Histogram{<:Real,1}, configs::FitConfigs)

    _check_spectrum(hist)

    edges = only(hist.edges)
    lower = configs.mu - configs.window_size / 2
    upper = configs.mu + configs.window_size / 2
    mask_edges = lower .<= edges .<= upper
    mask_bins = mask_edges[1:(end-1)] .& mask_edges[2:end]

    any(mask_bins) ||
        throw(ArgumentError("The fit window [$lower, $upper] keV contains no complete bin"))

    return Histogram(edges[mask_edges], hist.weights[mask_bins], hist.closed)

end

"""
    mean_background(weights::AbstractVector{<:Real}, bin_size::Real) -> Real

Estimate the mean background level in counts/keV from observed per-bin counts.

The estimate is floored at one count per bin so that the `Uniform` prior bounds constructed 
in [`build_prior`](@ref) stay valid, regardless of whether `weights` covers the full data 
range or only the bins outside a peak region.

# Arguments
- `weights::AbstractVector{<:Real}`: observed counts per bin
- `bin_size::Real`: width of one bin in keV

# Returns
- The mean background in counts/keV

# See also
- [`get_peak_features`](@ref) for peak-feature estimation
- [`build_prior`](@ref) for the prior that consumes the estimate
"""
function mean_background(weights::AbstractVector{<:Real}, bin_size::Real)

    return max(mean(weights), 1.0) / bin_size

end

"""
    background_poly_bounds(mean_background::Real, bin_size::Real, configs::FitConfigs)

Compute symmetric prior bounds for the linear and quadratic background coefficients.

The polynomial background is expanded around the centroid `configs.mu`,

```math
f_{\\text{bg}}(x) = C_q\\,(x - \\mu)^2 + C_l\\,(x - \\mu) + C_c,
```

so the coefficients `C_l` (counts/keV²) and `C_q` (counts/keV³) are dimensionful. Their 
natural scale is set by the mean background level `mean_background` (counts/keV) and the 
half-width of the fit window `h = configs.window_size / 2` (keV): the bounds are chosen so 
that each term can vary by up to twice the background level across the window, giving

```math
|C_l| \\le \\frac{2\\,\\text{mean\\_background}}{h}, \\qquad
|C_q| \\le \\frac{2\\,\\text{mean\\_background}}{h^2}.
```

# Arguments
- `mean_background::Real`: estimated mean background level in counts/keV
- `bin_size::Real`: bin width in keV, the lower limit of `h`
- `configs::FitConfigs`: fitting configuration providing `mu` and `window_size`

# Returns
- A `Tuple` `(lin_bound, quad_bound)` with the positive bounds for `C_l` and `C_q`

# See also
- [`mean_background`](@ref) for the background-level estimate
- [`build_prior`](@ref) which consumes these bounds
"""
function background_poly_bounds(mean_background::Real, bin_size::Real, configs::FitConfigs)

    half_window = configs.window_size / 2
    half_window = max(half_window, bin_size)

    lin_bound = 2 * mean_background / half_window
    quad_bound = 2 * mean_background / half_window^2

    return lin_bound, quad_bound

end

"""
    get_peak_features(hist::Histogram{<:Real,1}, configs::FitConfigs)

Estimate the peak height and area from observed count data.

Bins with their centers within `+-3 * configs.sigma` of the centroid `configs.mu` are
identified as the peak region. First, the mean background is calculated from the bins
outside peak region. Then the height is taken as the maximum observed count in the peak
area minus the mean background. The peak area is estimated as
`sqrt(2 * pi) * configs.sigma * peak_height`.

Poisson data can produce zero background counts or no peak above the background, so 
`peak_height` and `mean_background` are floored at one count per bin; `peak_area` is 
derived from the floored height. This keeps the `Uniform` prior bounds constructed in 
[`build_prior`](@ref) valid.

# Arguments
- `hist::Histogram{<:Real,1}`: binned spectrum, see [`_check_spectrum`](@ref)
- `configs::FitConfigs`: fitting configuration providing the centroid `mu` and the width
  `sigma` (`sigma > 0`)

# Returns
- A tuple `(peak_height, peak_area, mean_background)` containing the estimated height and 
  area of the peak, as well as the estimated mean background in 
  (counts/keV, counts, counts/keV)

# Throws
- The `ArgumentError`s of [`_check_spectrum`](@ref)
- An `ArgumentError` if `sigma` is not positive
- An `ArgumentError` if `mu +- 3 * sigma` is not contained within the range of the bin
  centers
- An `ArgumentError` if the peak region contains no bin centers
- An `ArgumentError` if only the peak region is contained within the range of the bin
  centers

# See also
- [`FitConfigs`](@ref) for the configuration options
- [`build_prior`](@ref) which uses these estimates for prior construction
"""
function get_peak_features(hist::Histogram{<:Real,1}, configs::FitConfigs)

    bin_size = _check_spectrum(hist)
    bin_centers = midpoints(only(hist.edges))
    weights = hist.weights

    mu = configs.mu
    sigma = configs.sigma

    sigma > 0 || throw(ArgumentError("`sigma` must be positive, got $sigma"))

    lower_peak_limit = mu - 3 * sigma
    upper_peak_limit = mu + 3 * sigma
    data_min = minimum(bin_centers)
    data_max = maximum(bin_centers)
    if lower_peak_limit < data_min || upper_peak_limit > data_max
        throw(
            ArgumentError(
                "Peak region [$lower_peak_limit, $upper_peak_limit] keV is not fully contained in the data range [$data_min, $data_max] keV.",
            ),
        )
    end

    peak_mask = lower_peak_limit .<= bin_centers .<= upper_peak_limit
    peak_weights = weights[peak_mask]                                   # counts/bin

    if isempty(peak_weights)
        throw(
            ArgumentError(
                "Peak region [$lower_peak_limit, $upper_peak_limit] keV contains no bin centers. Use a larger `sigma` or supply data with finer bins.",
            ),
        )
    end

    if length(peak_weights) >= length(weights)
        throw(
            ArgumentError(
                "Data range [$data_min, $data_max] keV contains only the peak region [$lower_peak_limit, $upper_peak_limit] keV. Need more data for background estimation.",
            ),
        )
    end

    background_weights = weights[.!peak_mask]                           # counts/bin
    mean_background_val = mean_background(background_weights, bin_size) # counts/keV
    peak_height =
        max(maximum(peak_weights) / bin_size - mean_background_val, 1.0 / bin_size)
    # counts/keV
    peak_area = sqrt(2 * pi) * sigma * peak_height                      # counts

    return peak_height, peak_area, mean_background_val

end

"""
    is_present(component::AbstractComponent)

Return `true` if `component` is part of the model (a concrete parameter struct, `Enabled`,
or a container) and `false` if it is `Disabled`.

Used for spec-time decisions such as which priors to build in [`build_prior`](@ref).

# Arguments
- `component::AbstractComponent`: the component (or container) to test

# Returns
- `true` if `component` is present (`Enabled`, a concrete parameter struct, or a container)
- `false` if `component` is `Disabled`

# See also
- [`AbstractComponent`](@ref), [`Enabled`](@ref), [`Disabled`](@ref) for the component  
  management
"""
is_present(::Disabled) = false
is_present(::AbstractComponent) = true
