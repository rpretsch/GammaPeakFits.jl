"""
    expected_counts(model::AbstractMeasure, edges::AbstractVector{<:Real}) -> Vector
    expected_counts(model::AbstractMeasure, hist::Histogram{<:Real,1}) -> Vector

Expected counts per bin of a spectral model, the mass of `model` in each bin. Exact for
the LegendSpecFits shapes and their sums and multiples (differences of the Stieltjes
measure function), Simpson's rule for measures without one.

The `edges` method accepts any strictly increasing edges, also non-uniform ones. The
`hist` method uses the bins of a spectrum checked by [`_check_spectrum`](@ref); its
counts are ignored.

# Arguments
- `model::AbstractMeasure`: spectral model, e.g.
  `n * GaussPeak(mu, sigma) + PolynomialShape(b)`
- `edges::AbstractVector{<:Real}`: bin edges in keV
- `hist::Histogram{<:Real,1}`: binned spectrum whose bins are used

# Throws
- An `ArgumentError` if `edges` has fewer than two entries or is not strictly increasing
- The `ArgumentError`s of [`_check_spectrum`](@ref) (`hist` method)

# Returns
- A vector with the expected counts per bin
"""
function expected_counts(model::AbstractMeasure, edges::AbstractVector{<:Real})

    length(edges) >= 2 || throw(ArgumentError("`edges` must contain at least two edges"))
    all(>(0), diff(edges)) || throw(ArgumentError("`edges` must be strictly increasing"))

    # Private LegendSpecFits API, called only here:
    return LegendSpecFits._bin_masses(model, edges)

end

function expected_counts(model::AbstractMeasure, hist::Histogram{<:Real,1})

    _check_spectrum(hist)

    return expected_counts(model, only(hist.edges))

end
