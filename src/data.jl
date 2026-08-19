"""
    SpectrumData

Container for binned energy spectrum data.

# Fields
- `bin_centers::Vector{Float64}`: bin centers in keV
- `bin_edges::Vector{Float64}`: bin edges in keV
- `weights::Vector{Int}`: observed counts per bin
- `bin_size::Float64`: width of each bin in keV

# Constructors

    SpectrumData(
        bin_markers::Vector{Float64},
        weights::Vector{Int};
        bin_size::Union{Float64,Nothing} = nothing,
    )

Construct a `SpectrumData` from bin markers and observed counts.

`bin_markers` are either the bin edges (`length(bin_markers) == length(weights) + 1`) or 
the bin centers (`length(bin_markers) == length(weights)`). The missing set is derived 
using `bin_size`. If `bin_size` is not supplied, it is estimated from the uniform bin 
markers.

# Arguments
- `bin_markers::Vector{Float64}`: bin edges or bin centers in keV
- `weights::Vector{Int}`: observed counts per bin
- `bin_size::Union{Float64,Nothing}`: width of each bin in keV (optional). If `nothing`, 
  the bin size is estimated from `bin_markers`

# Throws
- An `ArgumentError` if `length(bin_markers)` is neither `length(weights)` nor
  `length(weights) + 1`
- An `ArgumentError` if `bin_size` is not supplied and only a single bin marker is given,
  since the bin size cannot be estimated
- An `ArgumentError` if `bin_size` is not supplied and the bins are not evenly spaced

---

    SpectrumData(
        lower_limit::Float64, 
        upper_limit::Float64, 
        bin_size::Float64, 
        params::ModelParams,
    )

Generate synthetic spectrum data from a model over a uniform grid of 
`(lower_limit):bin_size:(upper_limit)`, then sampling Poisson-distributed counts for each 
bin.

# Arguments
- `lower_limit::Float64`: start of the energy range in keV
- `upper_limit::Float64`: end of the energy range in keV
- `bin_size::Float64`: width of each bin in keV
- `params::ModelParams`: model parameters used to compute expected counts

# Throws
- An `ArgumentError` if the model produces negative or non-finite expected counts

# Returns
- A `SpectrumData` object

# See also
- [`full_model`](@ref) for the used model
"""
Base.@kwdef struct SpectrumData
    bin_centers::Vector{Float64}
    bin_edges::Vector{Float64}
    weights::Vector{Int}
    bin_size::Float64
end

function SpectrumData(
    bin_markers::Vector{Float64},
    weights::Vector{Int};
    bin_size::Union{Float64,Nothing} = nothing,
)
    binMarker_length = length(bin_markers)
    weights_length = length(weights)

    if isnothing(bin_size)
        (binMarker_length == 1) && throw(
            ArgumentError(
                "Bin size can not be estimated, you must supply it directly via `bin_size`",
            ),
        )
        bin_sizes = bin_markers[2:end] - bin_markers[1:(end-1)]
        all(==(bin_sizes[1]), bin_sizes) ? bin_size = bin_sizes[1] :
        throw(ArgumentError("Your bins are not evenly spaced"))
    end

    if binMarker_length == weights_length
        bin_centers = bin_markers
        bin_edges =
            collect((bin_centers[1]-bin_size/2):bin_size:(bin_centers[end]+bin_size/2))
    elseif binMarker_length == (weights_length + 1)
        bin_edges = bin_markers
        bin_centers =
            collect((bin_edges[1]+bin_size/2):bin_size:(bin_edges[end]-bin_size/2))
    else
        throw(
            ArgumentError(
                "Your `bin_markers` and `weights` sizes do not match. Got $binMarker_length and $weights_length",
            ),
        )
    end

    return SpectrumData(
        bin_centers = bin_centers,
        bin_edges = bin_edges,
        weights = weights,
        bin_size = bin_size,
    )
end

function SpectrumData(
    lower_limit::Float64,
    upper_limit::Float64,
    bin_size::Float64,
    params::ModelParams,
)
    bin_centers = collect(range(lower_limit, upper_limit; step = bin_size))
    bin_edges =
        collect(range(lower_limit - bin_size/2, upper_limit + bin_size/2; step = bin_size))
    expected_counts = full_model(bin_centers, params) .* bin_size
    any(x -> !isfinite(x) || x<0, expected_counts) && throw(
        ArgumentError(
            "Model produced negative or non-finite expected counts; check parameters.",
        ),
    )
    weights = rand.(Poisson.(expected_counts))
    return SpectrumData(
        bin_centers = bin_centers,
        bin_edges = bin_edges,
        weights = weights,
        bin_size = bin_size,
    )
end
