"""
    cut_data(data::SpectrumData, configs::FitConfigs)

Slice a spectrum to the fit window centered on a peak.

Only bins that lie entirely within 
`[configs.mu - configs.window_size/2, configs.mu + configs.window_size/2]` are retained.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `configs::FitConfigs`: fitting configuration providing the centroid `mu` and the full 
  window width `window_size`

# Returns
- A new `SpectrumData` containing only the bins in the selected window

# See also
- [`FitConfigs`](@ref) for the configuration options
- [`SpectrumData`](@ref) for the data struct
"""
function cut_data(data::SpectrumData, configs::FitConfigs)

    mu = configs.mu
    window_size = configs.window_size

    mask_edges = (mu - window_size/2) .<= data.bin_edges .<= (mu + window_size/2)
    mask_centers = mask_edges[1:(end-1)] .& mask_edges[2:end]

    return SpectrumData(
        bin_centers = data.bin_centers[mask_centers],
        bin_edges = data.bin_edges[mask_edges],
        weights = data.weights[mask_centers],
        bin_size = data.bin_size,
    )
end

"""
    _mean_background(weights::Vector{Int}, bin_size::Float64)

Estimate the mean background level in counts/keV from observed per-bin counts.

The estimate is floored at one count per bin so that the `Uniform` prior bounds constructed 
in [`build_prior`](@ref) stay valid, regardless of whether `weights` covers the full data 
range or only the bins outside a peak region.

# Arguments
- `weights::Vector{Int}`: observed counts per bin
- `bin_size::Float64`: width of one bin in keV

# Returns
- The mean background in counts/keV

# See also
- [`get_peak_features`](@ref) for peak-feature estimation
- [`build_prior`](@ref) for the prior that consumes the estimate
"""
_mean_background(weights::Vector{Int}, bin_size::Float64) =
    max(mean(weights), 1.0) / bin_size

"""
    get_peak_features(data::SpectrumData, configs::FitConfigs)

Estimate the peak height and area from observed count data.

Bins within `+-3 * configs.sigma` of the centroid `configs.mu` are identified as the peak 
region. First, the mean background is calculated from the bins outside peak region. Then 
the height is taken as the maximum observed count in the peak area minus the mean 
background. The peak area is estimated as `sqrt(2 * pi) * configs.sigma * peak_height`.

Poisson data can produce zero background counts or no peak above the background, so 
`peak_height` and `mean_background` are floored at one count per bin; `peak_area` is 
derived from the floored height. This keeps the `Uniform` prior bounds constructed in 
[`build_prior`](@ref) valid.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `configs::FitConfigs`: fitting configuration providing the centroid `mu` and the width 
  `sigma` (`sigma > 0`)

# Returns
- A tuple `(peak_height, peak_area, mean_background)` containing the estimated height and 
  area of the peak, as well as the estimated mean background in 
  (counts/keV, counts, counts/keV)

# Throws
- An `ArgumentError` if `sigma` is not positive
- An `ArgumentError` if `mu +- 3 * sigma` is not contained within the range of 
  `data.bin_centers`
- An `ArgumentError` if the peak region contains no bin centers
- An `ArgumentError` if only the peak region is contained within the range of 
  `data.bin_centers`

# See also
- [`FitConfigs`](@ref) for the configuration options
- [`SpectrumData`](@ref) for the data struct
- [`build_prior`](@ref) which uses these estimates for prior construction
"""
function get_peak_features(data::SpectrumData, configs::FitConfigs)

    mu = configs.mu
    sigma = configs.sigma

    sigma > 0 || throw(ArgumentError("`sigma` must be positive, got $sigma"))

    lower_peak_limit = mu - 3 * sigma
    upper_peak_limit = mu + 3 * sigma
    data_min = minimum(data.bin_centers)
    data_max = maximum(data.bin_centers)
    if lower_peak_limit < data_min || upper_peak_limit > data_max
        throw(
            ArgumentError(
                "Peak region [$lower_peak_limit, $upper_peak_limit] keV is not fully contained in the data range [$data_min, $data_max] keV.",
            ),
        )
    end

    peak_mask = lower_peak_limit .<= data.bin_centers .<= upper_peak_limit
    peak_weights = data.weights[peak_mask]                  # counts/bin

    if isempty(peak_weights)
        throw(
            ArgumentError(
                "Peak region [$lower_peak_limit, $upper_peak_limit] keV contains no bin centers. Use a larger `sigma` or supply data with finer bins.",
            ),
        )
    end

    if length(peak_weights) >= length(data.weights)
        throw(
            ArgumentError(
                "Data range [$data_min, $data_max] keV contains only the peak region [$lower_peak_limit, $upper_peak_limit] keV. Need more data for background estimation.",
            ),
        )
    end

    background_weights = data.weights[.!peak_mask]                          # counts/bin
    mean_background = _mean_background(background_weights, data.bin_size)   # counts/keV
    peak_height =
        max(maximum(peak_weights) / data.bin_size - mean_background, 1.0 / data.bin_size)
    # counts/keV
    peak_area = sqrt(2 * pi) * sigma * peak_height                          # counts

    return peak_height, peak_area, mean_background
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
