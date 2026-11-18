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
- An `ArgumentError` if the bins are not evenly spaced
- An `ArgumentError` if `bin_size` is supplied and does not match the spacing of
  `bin_markers`

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

    if binMarker_length > 1
        bin_sizes = bin_markers[2:end] - bin_markers[1:(end-1)]
        mean_bin_size = mean(bin_sizes)
        all(≈(mean_bin_size), bin_sizes) ||
            throw(ArgumentError("Your bins are not evenly spaced"))
        isnothing(bin_size) ||
            bin_size ≈ mean_bin_size ||
            throw(
                ArgumentError(
                    "`bin_size` $bin_size does not match the bin spacing $mean_bin_size",
                ),
            )
        bin_size = something(bin_size, mean_bin_size)
    elseif isnothing(bin_size)
        throw(
            ArgumentError(
                "Bin size can not be estimated, you must supply it directly via `bin_size`",
            ),
        )
    end

    if binMarker_length == weights_length
        bin_centers = bin_markers
        bin_edges = [bin_centers .- bin_size/2; bin_centers[end] + bin_size/2]
    elseif binMarker_length == (weights_length + 1)
        bin_edges = bin_markers
        bin_centers = (bin_edges[1:(end-1)] .+ bin_edges[2:end]) ./ 2
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

"""
    bin_energies(
        energies::AbstractVector{<:Real},
        bin_size::Real;
        energy_range::Union{Tuple{Real,Real},Nothing} = nothing,
    ) -> SpectrumData

Histogram single energies, e.g. of detector hits, into bins of `bin_size`.

# Arguments
- `energies::AbstractVector{<:Real}`: energies in keV
- `bin_size::Real`: bin width in keV

# Keyword Arguments
- `energy_range::Union{Tuple{Real,Real},Nothing} = nothing`: lower edge of the first and
  upper edge of the last bin in keV; energies outside are dropped. If `nothing`, the bins
  cover all energies, with edges at whole multiples of `bin_size`

# Throws
- An `ArgumentError` if `bin_size` is not positive
- An `ArgumentError` if `energy_range` is not a whole multiple of `bin_size` wide
- An `ArgumentError` if `energies` is empty and no `energy_range` is given

# Returns
- A `SpectrumData` object with the counts per bin
"""
function bin_energies(
    energies::AbstractVector{<:Real},
    bin_size::Real;
    energy_range::Union{Tuple{Real,Real},Nothing} = nothing,
)
    bin_size > 0 || throw(ArgumentError("`bin_size` must be positive, got $bin_size"))

    if isnothing(energy_range)
        isempty(energies) && throw(
            ArgumentError("No energies to derive the range from, supply `energy_range`"),
        )
        first_bin = floor(Int, minimum(energies) / bin_size)
        last_bin = floor(Int, maximum(energies) / bin_size)
        bin_edges = collect(Float64, (first_bin:(last_bin+1)) .* bin_size)
    else
        lower, upper = energy_range
        nbins = round(Int, (upper - lower) / bin_size)
        nbins >= 1 && nbins * bin_size ≈ upper - lower || throw(
            ArgumentError(
                "`energy_range` $energy_range is no whole multiple of $bin_size wide",
            ),
        )
        bin_edges = collect(Float64, lower .+ (0:nbins) .* bin_size)
    end

    weights = fit(Histogram, energies, bin_edges).weights
    return SpectrumData(bin_edges, weights; bin_size = Float64(bin_size))
end

"""
    rebin(data::SpectrumData, bin_size::Real) -> SpectrumData

Merge neighbouring bins of `data` into bins of `bin_size`; an incomplete group of bins at
the upper end is dropped.

# Arguments
- `data::SpectrumData`: binned spectrum
- `bin_size::Real`: new bin width in keV, a whole multiple of `data.bin_size`

# Throws
- An `ArgumentError` if `bin_size` is no whole multiple of `data.bin_size`
- An `ArgumentError` if `bin_size` is wider than the whole spectrum

# Returns
- A `SpectrumData` object with the counts per new bin
"""
function rebin(data::SpectrumData, bin_size::Real)
    factor = round(Int, bin_size / data.bin_size)
    factor >= 1 && factor * data.bin_size ≈ bin_size || throw(
        ArgumentError("`bin_size` $bin_size is no whole multiple of $(data.bin_size)"),
    )
    nbins = length(data.weights) ÷ factor
    nbins >= 1 || throw(ArgumentError("`bin_size` $bin_size is wider than the spectrum"))
    weights = vec(sum(reshape(data.weights[1:(nbins*factor)], factor, nbins); dims = 1))

    return SpectrumData(
        data.bin_edges[1:factor:(nbins*factor+1)],
        weights;
        bin_size = Float64(bin_size),
    )
end
