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

Bins within `+-3 * sigma` of the centroid are identified as the peak region. The peak 
height is taken as the maximum observed count in that window. The peak area is estimated as
`6 * sigma * peak_height`. Both values are converted from counts/bin to counts/keV by 
dividing by`bin_size`.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `mu::Float64`: estimated centroid position in keV
- `sigma::Float64`: estimated standard deviation in keV

# Returns
- A tuple `(peak_height, peak_area)` containing the estimated height and area of the peak,
  in (counts/keV, counts)

# Throws
- An `ArgumentError` if `mu +- 3 * sigma` is not contained within the range of 
  `data.bin_centers`

# See also
- [`SpectrumData`](@ref) for the data struct
- [`build_prior`](@ref) which uses these estimates for prior construction
"""
function get_peak_features(data::SpectrumData, mu::Float64, sigma::Float64)

    lower = mu - 3 * sigma
    upper = mu + 3 * sigma
    data_min = minimum(data.bin_centers)
    data_max = maximum(data.bin_centers)
    if lower < data_min || upper > data_max
        throw(
            ArgumentError(
                "Peak region [$lower, $upper] keV is not fully contained in the data range [$data_min, $data_max] keV.",
            ),
        )
    end

    peak_mask = lower .<= data.bin_centers .<= upper
    peak_height = maximum(data.weights[peak_mask])  # counts/bin
    peak_area = 6 * sigma * peak_height             # counts/bin * keV

    peak_height_kev = peak_height / data.bin_size   # counts/keV
    peak_area_keV = peak_area / data.bin_size       # counts

    return peak_height_kev, peak_area_keV
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
