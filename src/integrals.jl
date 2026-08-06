"""
    numerical_integral(data::SpectrumData, params::ModelParams)

Integrate the [`full_model`](@ref) numerically over each energy bin using `QuadGK.quadgk`.

Skips model components that were set to `false`.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::ModelParams`: model parameters

# Returns
- An array of expected counts per bin

# See also
- [`full_model`](@ref) for the integrated model
- [`ModelParams`](@ref) for the model params
"""
function numerical_integral(data::SpectrumData, params::ModelParams)
    return first.(
        quadgk.(
            x -> full_model(x, params),
            data.bin_centers .- data.bin_size/2,
            data.bin_centers .+ data.bin_size/2,
        ),
    )
end

"""
    analytical_integral(data::SpectrumData, params::ModelParams)

Integrate the full model analytically over each energy bin.

Skips model components that were set to `false`.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::ModelParams`: model parameters

# Returns
- An array of expected counts per bin

# See also
- [`gaussian_integral`](@ref), [`compton_integral`](@ref), [`exGaussian_integral`](@ref),
  [`quadPoly_integral`](@ref), [`linPoly_integral`](@ref), and [`constPoly_integral`](@ref)
  for the individual components
- [`ModelParams`](@ref) for the model params
"""
function analytical_integral(data::SpectrumData, params::ModelParams)
    peak = params.peak
    background = params.background
    result = zeros(eltype(data.bin_centers), length(data.bin_centers))
    if !isnothing(peak)
        (peak.gaussian !== false) && (result .+= gaussian_integral(data, peak.gaussian))
        (peak.compton !== false) && (result .+= compton_integral(data, peak.compton))
        (peak.lowEnergyTail !== false) &&
            (result .+= exGaussian_integral(data, peak.lowEnergyTail))
        (peak.highEnergyTail !== false) &&
            (result .+= exGaussian_integral(data, peak.highEnergyTail))
    end
    if !isnothing(background)
        (background.quadPoly !== false) &&
            (result .+= quadPoly_integral(data, background.quadPoly))
        (background.linPoly !== false) &&
            (result .+= linPoly_integral(data, background.linPoly))
        (background.constPoly !== false) &&
            (result .+= constPoly_integral(data, background.constPoly))
    end
    return result
end

"""
    gaussian_integral(data::SpectrumData, params::GaussianParams)

Integrate the scaled Gaussian component analytically over each energy bin.

Uses `SpecialFunctions.erf`.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::GaussianParams`: Gaussian component parameters

# Returns
- An array of expected Gaussian counts per bin

# See also
- [`gaussian`](@ref) for the pointwise model
- [`GaussianParams`](@ref) for the parameters
"""
function gaussian_integral(data::SpectrumData, params::GaussianParams)
    return @. params.A/2 * (
        erf((data.bin_edges[2:end] - params.mu)/(sqrt(2) * params.sigma)) -
        erf((data.bin_edges[1:(end-1)] - params.mu)/(sqrt(2) * params.sigma))
    )
end

"""
    compton_integral(data::SpectrumData, params::ComptonParams)

Integrate the Compton-edge step function component analytically over each energy bin.

!!! warning "Not implemented"
    TODO: this function has not been implemented yet.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::ComptonParams`: Compton component parameters

# Returns
- An array of expected Compton counts per bin

# See also
- [`compton`](@ref) for the pointwise model
- [`ComptonParams`](@ref) for the parameters
"""
function compton_integral(data::SpectrumData, params::ComptonParams)
    return # TODO
end

"""
    exGaussian_integral(data::SpectrumData, params::ExGaussianParams)

Integrate the ex-Gaussian tail component analytically over each energy bin.

!!! warning "Not implemented"
    TODO: this function has not been implemented yet.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::ExGaussianParams`: tail component parameters

# Returns
- An array of expected tail counts per bin

# See also
- [`exGaussian`](@ref) for the pointwise model
- [`ExGaussianParams`](@ref) for the parameters
"""
function exGaussian_integral(data::SpectrumData, params::ExGaussianParams)
    return # TODO
end

"""
    quadPoly_integral(data::SpectrumData, params::QuadPolyParams)

Integrate the quadratic polynomial component analytically over each energy bin.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::QuadPolyParams`: quadratic polynomial parameters

# Returns
- An array of expected quadratic background counts per bin

# See also
- [`quad_polynomial`](@ref) for the pointwise model
- [`QuadPolyParams`](@ref) for the parameters
"""
function quadPoly_integral(data::SpectrumData, params::QuadPolyParams)
    return @. params.C/3 * (
        (data.bin_edges[2:end] - params.mu)^3 - (data.bin_edges[1:(end-1)] - params.mu)^3
    )
end

"""
    linPoly_integral(data::SpectrumData, params::LinPolyParams)

Integrate the linear polynomial component analytically over each energy bin.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::LinPolyParams`: linear polynomial parameters

# Returns
- An array of expected linear background counts per bin

# See also
- [`lin_polynomial`](@ref) for the pointwise model
- [`LinPolyParams`](@ref) for the parameters
"""
function linPoly_integral(data::SpectrumData, params::LinPolyParams)
    return @. params.C/2 * (
        data.bin_edges[2:end] * (data.bin_edges[2:end] - 2 * params.mu) -
        data.bin_edges[1:(end-1)] * (data.bin_edges[1:(end-1)] - 2 * params.mu)
    )
end

"""
    constPoly_integral(data::SpectrumData, params::ConstPolyParams)

Integrate the constant polynomial component analytically over each energy bin.

# Arguments
- `data::SpectrumData`: binned spectrum data
- `params::ConstPolyParams`: constant polynomial parameters

# Returns
- An array of expected constant background counts per bin

# See also
- [`const_polynomial`](@ref) for the pointwise model
- [`ConstPolyParams`](@ref) for the parameters
"""
function constPoly_integral(data::SpectrumData, params::ConstPolyParams)
    return @. params.C * (data.bin_edges[2:end] - data.bin_edges[1:(end-1)])
end
