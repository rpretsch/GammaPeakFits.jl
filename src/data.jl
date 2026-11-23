"""
    _check_edges(edges::AbstractVector{<:Real}) -> Real

Check that `edges` are valid bin edges: at least two entries, strictly increasing and
evenly spaced. The spacing is compared with `≈`, so that rounded float edges are accepted.

# Arguments
- `edges::AbstractVector{<:Real}`: bin edges in keV

# Throws
- An `ArgumentError` if `edges` has fewer than two entries
- An `ArgumentError` if `edges` is not strictly increasing
- An `ArgumentError` if the edges are not evenly spaced

# Returns
- The bin width in keV (in the number type of the edges), the mean spacing of the edges
"""
function _check_edges(edges::AbstractVector{<:Real})

    length(edges) >= 2 || throw(ArgumentError("Bin edges need at least two entries"))

    bin_sizes = diff(edges)
    all(>(0), bin_sizes) || throw(ArgumentError("Bin edges must be strictly increasing"))

    bin_size = mean(bin_sizes)
    all(≈(bin_size), bin_sizes) || throw(ArgumentError("Bin edges are not evenly spaced"))

    return bin_size

end

"""
    _check_spectrum(hist::Histogram{<:Real,1}) -> Real

Check that `hist` is a binned spectrum the fit functions can use: valid bin edges (see
[`_check_edges`](@ref)) and counts per bin (no density) that are non-negative whole
numbers.

# Arguments
- `hist::Histogram{<:Real,1}`: one-dimensional histogram of counts per bin

# Throws
- The `ArgumentError`s of [`_check_edges`](@ref)
- An `ArgumentError` if `hist` is a density
- An `ArgumentError` if a weight is no non-negative whole number

# Returns
- The bin width in keV (in the number type of the edges), the mean spacing of the bin edges
"""
function _check_spectrum(hist::Histogram{<:Real,1})

    bin_size = _check_edges(only(hist.edges))

    hist.isdensity && throw(ArgumentError("`hist` is a density, counts per bin are needed"))

    all(w -> isinteger(w) && w >= 0, hist.weights) || throw(
        ArgumentError("`hist` has weights that are no counts (non-negative whole numbers)"),
    )

    return bin_size

end

"""
    _uniform_edges(lower::Real, upper::Real, bin_size::Real)

Return the edges of uniform bins of `bin_size` from `lower` to `upper`. The edges are
computed from integer bin indices as `lower + i * bin_size`, so that rounding neither
accumulates over the bins nor changes their number, as it can for a float range.

# Arguments
- `lower::Real`: lower edge of the first bin in keV
- `upper::Real`: upper edge of the last bin in keV
- `bin_size::Real`: bin width in keV

# Throws
- An `ArgumentError` if `bin_size` is not positive
- An `ArgumentError` if `upper - lower` is not a positive whole multiple of `bin_size`

# Returns
- A `Vector{Float64}` of `n + 1` edges for `n = (upper - lower) / bin_size` bins

# See also
- [`bin_energies`](@ref), [`sample_spectrum`](@ref) for the users
"""
function _uniform_edges(lower::Real, upper::Real, bin_size::Real)
    bin_size > 0 || throw(ArgumentError("`bin_size` must be positive, got $bin_size"))
    nbins = round(Int, (upper - lower) / bin_size)
    nbins >= 1 && nbins * bin_size ≈ upper - lower || throw(
        ArgumentError(
            "Energy range ($lower, $upper) is no whole multiple of $bin_size wide",
        ),
    )
    return collect(Float64, lower .+ (0:nbins) .* bin_size)
end

"""
    bin_energies(
        energies::AbstractVector{<:Real},
        bin_size::Real;
        energy_range::Union{Tuple{Real,Real},Nothing} = nothing,
    ) -> Histogram

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
- A `Histogram` with the counts per bin
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
        bin_edges = _uniform_edges(energy_range..., bin_size)
    end

    return fit(Histogram, energies, bin_edges)

end

"""
    rebin(hist::Histogram{<:Real,1}, bin_size::Real) -> Histogram

Merge neighbouring bins of `hist` into bins of `bin_size`; an incomplete group of bins at 
the upper end is dropped.

# Arguments
- `hist::Histogram{<:Real,1}`: binned spectrum, see [`_check_spectrum`](@ref)
- `bin_size::Real`: new bin width in keV, a whole multiple of the bin width of `hist`

# Throws
- The `ArgumentError`s of [`_check_spectrum`](@ref)
- An `ArgumentError` if `bin_size` is no whole multiple of the bin width of `hist`
- An `ArgumentError` if `bin_size` is wider than the whole spectrum

# Returns
- A `Histogram` with the counts per new bin
"""
function rebin(hist::Histogram{<:Real,1}, bin_size::Real)

    old_bin_size = _check_spectrum(hist)

    factor = round(Int, bin_size / old_bin_size)
    factor >= 1 && factor * old_bin_size ≈ bin_size ||
        throw(ArgumentError("`bin_size` $bin_size is no whole multiple of $old_bin_size"))

    nbins = length(hist.weights) ÷ factor
    nbins >= 1 || throw(ArgumentError("`bin_size` $bin_size is wider than the spectrum"))

    weights = vec(sum(reshape(hist.weights[1:(nbins*factor)], factor, nbins); dims = 1))

    return Histogram(only(hist.edges)[1:factor:(nbins*factor+1)], weights, hist.closed)

end

"""
    sample_spectrum(model::AbstractMeasure, edges::AbstractVector{<:Real}) -> Histogram
    sample_spectrum(
        model::AbstractMeasure;
        energy_range::Tuple{Real,Real},
        bin_size::Real,
    ) -> Histogram

Draw a synthetic spectrum from a measure model: Poisson-distributed counts around the
expected counts per bin from [`expected_counts`](@ref).

The `edges` method samples on given bin edges, e.g. `only(hist.edges)` to reproduce the
binning of a measured spectrum. The bins must be uniform, so that the result is a valid
spectrum for the fit functions (see [`_check_spectrum`](@ref)). The keyword method builds
uniform edges from `bin_size` and `energy_range`.

# Arguments
- `model::AbstractMeasure`: spectral model, e.g.
  `n * GaussPeak(mu, sigma) + PolynomialShape(b)`
- `edges::AbstractVector{<:Real}`: uniform, increasing bin edges in keV

# Keyword Arguments
- `energy_range::Tuple{Real,Real}`: lower edge of the first and upper edge of the last bin
  in keV
- `bin_size::Real`: bin width in keV

# Throws
- The `ArgumentError`s of [`expected_counts`](@ref) and [`_check_spectrum`](@ref)
  (`edges` method)
- The `ArgumentError`s of [`_uniform_edges`](@ref) (keyword method)
- An `ArgumentError` if the model produces negative or non-finite expected counts

# Returns
- A `Histogram` with the sampled counts per bin
"""
function sample_spectrum(model::AbstractMeasure, edges::AbstractVector{<:Real})

    counts = expected_counts(model, edges)

    any(x -> !isfinite(x) || x < 0, counts) && throw(
        ArgumentError(
            "Model produced negative or non-finite expected counts; check parameters.",
        ),
    )

    hist = Histogram(edges, rand.(Poisson.(counts)))
    _check_spectrum(hist)  # enforces uniform bins, as the result is a spectrum

    return hist

end

function sample_spectrum(
    model::AbstractMeasure;
    bin_size::Real,
    energy_range::Tuple{Real,Real},
)

    return sample_spectrum(model, _uniform_edges(energy_range..., bin_size))

end
