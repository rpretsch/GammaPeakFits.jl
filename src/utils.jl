"""
    plot_data(data::SpectrumData, mu::Union{<:AbstractFloat,Nothing} = nothing)

Plot a binned energy spectrum as a bar plot with optional marker for the peak centroid.

# Arguments
- `data::SpectrumData`: binned spectrum data to plot
- `mu::Union{AbstractFloat,Nothing}`: optional centroid position to highlight with a 
  vertical line

# Returns
- A tuple `(fig, ax)` of a `Makie.Figure` and `Makie.Axis`, suitable for further 
  customization or display.

# See also
- [`SpectrumData`](@ref) for the data struct
"""
function plot_data(data::SpectrumData, mu::Union{<:AbstractFloat,Nothing} = nothing)

    fig = Figure(size = (1800, 600))
    ax = Axis(fig[1, 1]; xlabel = "Energy [keV]", ylabel = "Counts", yscale = log10)

    barplot!(
        ax,
        data.bin_centers,
        data.weights;
        width = data.bin_size,
        gap = 0,
        strokewidth = 0,
    )
    !isnothing(mu) && vlines!(mu)

    return fig, ax
end

"""
    cut_data(data::SpectrumData, mu::AbstractFloat, window_size::AbstractFloat)

Slice a spectrum to a region of interest centered on a peak.

Only bins whose centers lie within `[mu - window_size/2, mu + window_size/2]` are retained.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `mu::AbstractFloat`: centroid position of the region in keV
- `window_size::AbstractFloat`: full width of the region in keV

# Returns
- A new `SpectrumData` containing only the bins in the selected window

# See also
- [`SpectrumData`](@ref) for the data struct
"""
function cut_data(data::SpectrumData, mu::AbstractFloat, window_size::AbstractFloat)

    mask = (mu - window_size/2) .<= data.bin_centers .<= (mu + window_size/2)

    return SpectrumData(
        bin_centers = data.bin_centers[mask],
        weights = data.weights[mask],
        bin_size = data.bin_size,
    )
end

"""
    get_peak_features(
        data::SpectrumData,
        mu::AbstractFloat,
        sigma::AbstractFloat,
    )

Estimate the peak height and area from observed count data.

Bins within `+-3 * sigma` of the centroid are identified as the peak region. The peak 
height is taken as the maximum observed count in that window. The peak area is estimated as
`6 * sigma * peak_height`. Both values are converted from counts/bin to counts/keV by 
dividing by`bin_size`.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `mu::AbstractFloat`: estimated centroid position in keV
- `sigma::AbstractFloat`: estimated standard deviation in keV

# Returns
- `(peak_height, peak_area) :: Tuple{Float64, Float64}` — both in counts/keV

# See also
- [`SpectrumData`](@ref) for the data struct
- [`build_prior`](@ref) which uses these estimates for prior construction
"""
function get_peak_features(data::SpectrumData, mu::AbstractFloat, sigma::AbstractFloat)

    # counts/bin
    peak_mask = (mu - 3 * sigma) .<= data.bin_centers .<= (mu + 3 * sigma)
    peak_height = maximum(data.weights[peak_mask])
    peak_area = 6 * sigma * peak_height

    # Convert to counts/keV
    peak_height_kev = peak_height / data.bin_size
    peak_area_keV = peak_area / data.bin_size

    return peak_height_kev, peak_area_keV
end
