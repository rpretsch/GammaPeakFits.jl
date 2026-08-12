"""
    cut_data(data::SpectrumData, mu::Float64, window_size::Float64)

Slice a spectrum to a region of interest centered on a peak.

Only bins that lie entirely within `[mu - window_size/2, mu + window_size/2]` are retained.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `mu::Float64`: centroid position of the region in keV
- `window_size::Float64`: full width of the region in keV

# Returns
- A new `SpectrumData` containing only the bins in the selected window

# See also
- [`SpectrumData`](@ref) for the data struct
"""
function cut_data(data::SpectrumData, mu::Float64, window_size::Float64)

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
    get_peak_features(data::SpectrumData, mu::Float64, sigma::Float64)

Estimate the peak height and area from observed count data.

Bins within `+-3 * sigma` of the centroid are identified as the peak region. First, the 
mean background is calculated from the bins outside peak region. Then the height is taken 
as the maximum observed count in the peak area minus the mean background. 
The peak area is estimated as `sqrt(2 * pi) * sigma * peak_height`.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `mu::Float64`: estimated centroid position in keV
- `sigma::Float64`: estimated standard deviation in keV

# Returns
- A tuple `(peak_height, peak_area, mean_background)` containing the estimated height and 
  area of the peak, as well as the estimated mean background in 
  (counts/keV, counts, counts/keV)

# Throws
- An `ArgumentError` if `mu +- 3 * sigma` is not contained within the range of 
  `data.bin_centers`
- An `ArgumentError` if only the peak region is contained within the range of 
  `data.bin_centers`

# See also
- [`SpectrumData`](@ref) for the data struct
- [`build_prior`](@ref) which uses these estimates for prior construction
"""
function get_peak_features(data::SpectrumData, mu::Float64, sigma::Float64)

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

    if length(peak_weights) >= length(data.weights)
        throw(
            ArgumentError(
                "Data range [$data_min, $data_max] keV contains only the peak region [$lower_peak_limit, $upper_peak_limit] keV. Need more data for background estimation.",
            ),
        )
    end

    background_weights = data.weights[.!peak_mask]          # counts/bin
    mean_background = mean(background_weights)              # counts/bin
    peak_height = maximum(peak_weights) - mean_background   # counts/bin
    peak_area = sqrt(2 * pi) * sigma * peak_height          # counts/bin * keV

    mean_background_keV = mean_background / data.bin_size   # counts/keV
    peak_height_kev = peak_height / data.bin_size           # counts/keV
    peak_area_keV = peak_area / data.bin_size               # counts

    return peak_height_kev, peak_area_keV, mean_background_keV
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
