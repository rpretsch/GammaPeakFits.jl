"""
    poisson_ll(bin_centers::Union{Real, AbstractVector{<:Real}}, weights::Union{Int, AbstractVector{<:Int}}, params::ModelParams, bin_size::Real)

Compute the Poisson log-likelihood for the model given observed counts.

The expected count in each bin is obtained by integrating [`full_model`](@ref) over the bin 
width `bin_size` using `QuadGK.jl`.
The log-probability of the observed integer count `weights` is then evaluated under a
Poisson distribution with that expected rate.

# Arguments
- `bin_centers::Union{Real, AbstractVector{<:Real}}`: bin center(s) in keV
- `weights::Union{Int, AbstractVector{<:Int}}`: observed count(s) per bin
- `params::ModelParams`: model parameters
- `bin_size::Real`: width of each bin (used for integration bounds) in keV

# Returns
- Scalar case: the log-likelihood for a single bin
- Vector case: sum of log-likelihoods across bins (total log-likelihood)

# See also
- [`full_model`](@ref) for the underlying model
- [`ModelParams`](@ref) for the parameter structure
"""
function poisson_ll(bin_centers::Real, weights::Int, params::ModelParams, bin_size::Real)
    expected_counts = quadgk(
        x -> full_model(x, params),
        bin_centers - bin_size/2,
        bin_centers + bin_size/2,
    )[1]
    return logpdf.(Poisson(expected_counts), weights)
end
function poisson_ll(
    bin_centers::AbstractVector{<:Real},
    weights::AbstractVector{<:Int},
    params::ModelParams,
    bin_size::Real,
)
    result_vector = poisson_ll.(bin_centers, weights, Ref(params), bin_size)
    return sum(result_vector)
end
