using BAT: PosteriorMeasure
using StatsBase: Histogram
using GammaPeakFits: _check_spectrum

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

    @testset "data checks" begin
        @test @inferred(_check_spectrum(HIST)) isa Real
    end

    @testset "measure models" begin
        @test @inferred(expected_counts(PolynomialShape(C_CONST), HIST)) isa Vector{Float64}
        @test @inferred(
            sample_spectrum(
                PolynomialShape(C_CONST);
                bin_size = BIN_SIZE,
                energy_range = ENERGY_RANGE,
            )
        ) isa Histogram
        @test @inferred(sample_spectrum(PolynomialShape(C_CONST), only(HIST.edges))) isa
              Histogram
    end

    @testset "build_posterior" begin
        configs = FitConfigs(mu = MU, sigma = SIGMA)
        pf_prior = build_prior(HIST, PF_KERNEL, configs)
        @test @inferred(build_posterior(HIST, PF_KERNEL, pf_prior)) isa PosteriorMeasure
    end

end
