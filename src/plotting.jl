"""
    plot_data(
        data::SpectrumData;
        mu::Union{Float64,Nothing} = nothing,
        yscale = log10,
        xlabel::AbstractString = "Energy [keV]",
        ylabel::AbstractString = "Counts",
        title::AbstractString = "",
        color = RGBf(0.36, 0.58, 0.78),
        figure_size = (1800, 600),
    )

Plot a binned energy spectrum as a bar plot with optional marker for the peak centroid.

# Arguments
- `data::SpectrumData`: binned spectrum data to plot

# Keyword arguments
- `mu::Union{Float64,Nothing}`: optional centroid position to highlight with a short
  dashed red vertical marker annotated in-line with "μ = `mu`` keV". Default: `nothing`
- `yscale`: y-axis scale, e.g. `log10` or `identity`. Default: `log10`
- `xlabel::AbstractString`: x-axis label. Default: `"Energy [keV]"`
- `ylabel::AbstractString`: y-axis label. Default: `"Counts"`
- `title::AbstractString`: axis title. Default: `""` (no title)
- `color`: bar colour (drawn without a stroke). Default: a muted blue
- `figure_size`: `Figure` size in pixels. Default: `(1800, 600)`

# Returns
- A tuple `(fig, ax)` of a `Makie.Figure` and `Makie.Axis`, suitable for further 
  customization or display.

# See also
- [`SpectrumData`](@ref) for the data struct
"""
function plot_data(
    data::SpectrumData;
    mu::Union{Float64,Nothing} = nothing,
    yscale = log10,
    xlabel::AbstractString = "Energy [keV]",
    ylabel::AbstractString = "Counts",
    title::AbstractString = "",
    color = RGBf(0.36, 0.58, 0.78),
    figure_size = (1800, 600),
)

    fig = Figure(size = figure_size)
    ax = Axis(
        fig[1, 1];
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        yscale = yscale,
        titlesize = 20,
        xlabelsize = 16,
        ylabelsize = 16,
        xticklabelsize = 13,
        yticklabelsize = 13,
        xgridcolor = (RGBf(0.5, 0.5, 0.5), 0.25),
        ygridcolor = (RGBf(0.5, 0.5, 0.5), 0.25),
        xgridwidth = 0.5,
        ygridwidth = 0.5,
    )

    barplot!(
        ax,
        data.bin_centers,
        data.weights;
        width = data.bin_size,
        gap = 0,
        color = color,
        strokewidth = 0,
    )
    if !isnothing(mu)
        vlines!(
            ax,
            mu;
            ymin = 0.01,
            ymax = 0.99,
            color = :red,
            linestyle = :dash,
            linewidth = 1.5,
        )
        label_y = lift(ax.finallimits) do lims
            lims.origin[2] + lims.widths[2]
        end
        text!(
            ax,
            mu,
            label_y;
            text = "μ = $(round(mu; digits = 1)) keV",
            color = :red,
            fontsize = 13,
            align = (:left, :top),
            offset = (4, -4),
            xautolimits = false,
            yautolimits = false,
        )
    end

    return fig, ax
end
