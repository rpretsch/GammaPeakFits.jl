using BAT: PosteriorMeasure
using StatsBase: Histogram, midpoints
using GammaPeakFits:
    _check_edges,
    _check_spectrum,
    _uniform_edges,
    _mean_background,
    _background_poly_bounds,
    _peak_mask,
    _get_peak_features

@testset "type stability" begin

    MU = 2048.0
    SIGMA = 10.0
    A = 1000.0
    C_CONST = 100.0
    BIN_SIZE = 1.0
    WINDOW = 100.0
    ENERGY_RANGE = (MU - WINDOW / 2 - BIN_SIZE / 2, MU + WINDOW / 2 + BIN_SIZE / 2)
    PF_KERNEL = @pf $n * GaussPeak($mu, $sigma) + PolynomialShape($b)

    HIST = sample_spectrum(
        A * GaussPeak(MU, SIGMA);
        bin_size = BIN_SIZE,
        energy_range = ENERGY_RANGE,
    )
    EDGES = only(HIST.edges)

    # _check_edges, _check_spectrum
    @test @inferred(_check_edges(EDGES)) isa Real
    @test @inferred(_check_spectrum(HIST)) isa Real

    # _uniform_edges
    @test @inferred(_uniform_edges(ENERGY_RANGE..., BIN_SIZE)) isa Vector{Float64}

    # bin_energies, derived and given energy range
    energies = [MU - SIGMA, MU, MU + SIGMA]
    @test @inferred(bin_energies(energies, BIN_SIZE)) isa Histogram
    @test @inferred(bin_energies(energies, BIN_SIZE; energy_range = ENERGY_RANGE)) isa
          Histogram

    # rebin, cut_data
    @test @inferred(rebin(HIST, 2 * BIN_SIZE)) isa Histogram
    @test @inferred(cut_data(HIST; center = MU, width = WINDOW / 2)) isa Histogram

    # expected_counts, edges and histogram method
    @test @inferred(expected_counts(PolynomialShape(C_CONST), EDGES)) isa Vector{Float64}
    @test @inferred(expected_counts(PolynomialShape(C_CONST), HIST)) isa Vector{Float64}

    # sample_spectrum, keyword and edges method
    @test @inferred(
        sample_spectrum(
            PolynomialShape(C_CONST);
            bin_size = BIN_SIZE,
            energy_range = ENERGY_RANGE,
        )
    ) isa Histogram
    @test @inferred(sample_spectrum(PolynomialShape(C_CONST), EDGES)) isa Histogram

    # _mean_background, _background_poly_bounds, _peak_mask, _get_peak_features
    @test @inferred(_mean_background(HIST.weights, BIN_SIZE)) isa Real
    @test @inferred(_background_poly_bounds(C_CONST, WINDOW / 2)) isa Tuple{Real,Real}
    @test @inferred(_peak_mask(midpoints(EDGES); mu = MU, sigma = SIGMA)) isa BitVector
    @test @inferred(_get_peak_features(HIST; mu = MU, sigma = SIGMA)) isa
          Tuple{Real,Real,Real}

    # peak_priors, background_priors without and with a peak
    @test @inferred(peak_priors(HIST; mu = MU, sigma = SIGMA)) isa NamedTuple
    @test @inferred(background_priors(HIST)) isa NamedTuple
    @test @inferred(background_priors(HIST; mu = MU, sigma = SIGMA)) isa NamedTuple

    # build_posterior (build_prior is not type-stable by design: the parameter names
    # are collected at runtime)
    defaults = merge(
        peak_priors(HIST; mu = MU, sigma = SIGMA),
        background_priors(HIST; mu = MU, sigma = SIGMA),
    )
    pf_prior = build_prior(PF_KERNEL, defaults)
    @test @inferred(build_posterior(HIST, PF_KERNEL, pf_prior)) isa PosteriorMeasure

end
