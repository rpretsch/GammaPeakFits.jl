"""
    expected_counts(model::AbstractMeasure, edges::AbstractVector{<:Real}) -> Vector
    expected_counts(model::AbstractMeasure, hist::Histogram{<:Real,1}) -> Vector

Expected counts per bin of a spectral model, the mass of `model` in each bin. Exact for
the LegendSpecFits shapes and their sums and multiples (differences of the Stieltjes
measure function), Simpson's rule for measures without one.

Both methods require uniform, increasing bins. The counts of the `hist` method are ignored.

# Arguments
- `model::AbstractMeasure`: spectral model, e.g.
  `n * GaussPeak(mu, sigma) + PolynomialShape(b)`
- `edges::AbstractVector{<:Real}`: bin edges in keV
- `hist::Histogram{<:Real,1}`: binned spectrum whose bins are used

# Throws
- The `ArgumentError`s of [`_check_edges`](@ref)

# Returns
- A vector with the expected counts per bin
"""
function expected_counts(model::AbstractMeasure, edges::AbstractVector{<:Real})

    _check_edges(edges)

    # Private LegendSpecFits API, called only here:
    return LegendSpecFits._bin_masses(model, edges)

end

function expected_counts(model::AbstractMeasure, hist::Histogram{<:Real,1})

    return expected_counts(model, only(hist.edges))

end
