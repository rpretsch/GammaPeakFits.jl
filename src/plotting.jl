"""
    plot_data(data::SpectrumData; mu::Union{Float64,Nothing} = nothing)

Plot a binned energy spectrum as a bar plot with optional marker for the peak centroid.

# Arguments
- `data::SpectrumData`: binned spectrum data to plot

# Keyword arguments
- `mu::Union{Float64,Nothing}`: optional centroid position to highlight with a 
  vertical line. Default: `nothing`

# Returns
- A tuple `(fig, ax)` of a `Makie.Figure` and `Makie.Axis`, suitable for further 
  customization or display.

# See also
- [`SpectrumData`](@ref) for the data struct
"""
function plot_data(data::SpectrumData; mu::Union{Float64,Nothing} = nothing)

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
    !isnothing(mu) && vlines!(ax, mu, color = :red)

    return fig, ax
end
