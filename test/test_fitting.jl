using BAT: PosteriorMeasure, distprod, logdensityof
using Distributions: Normal, Uniform, truncated
using GammaPeakFits: _mean_background, _background_poly_bounds, _get_peak_features
using StatsBase: Histogram

@testset "fitting" begin

    A = 1000.0
    MU = 2048.0
    SIGMA = 10.0
    WINDOW = 100.0
    BIN_SIZE = 1.0
    DATA = sample_spectrum(
        A * GaussPeak(MU, SIGMA);
        bin_size = BIN_SIZE,
        energy_range = (MU - WINDOW / 2 - BIN_SIZE / 2, MU + WINDOW / 2 + BIN_SIZE / 2),
    )
    PEAK_KERNEL = @pf $n * (
        (1 - $alpha) * GaussPeak($mu, $sigma) + $alpha * LowETailPeak($mu, $sigma, $theta)
    ) + PolynomialShape($b)
    NESTED_KERNEL = @pf $peak.n * GaussPeak($peak.mu, $peak.sigma) + PolynomialShape($bkg.b)
    DEFAULTS = (
        peak = peak_priors(DATA; mu = MU, sigma = SIGMA),
        bkg = background_priors(DATA; mu = MU, sigma = SIGMA),
    )

    @testset "default blocks" begin

        @testset "peak_priors" begin
            priors = peak_priors(DATA; mu = MU, sigma = SIGMA)
            _, peak_area, _ = _get_peak_features(DATA; mu = MU, sigma = SIGMA)

            @test keys(priors) == (:n, :alpha, :mu, :sigma, :theta)
            @test priors.n == Uniform(0, 3 * peak_area)
            @test priors.alpha == Uniform(0, 1)
            @test priors.mu == Normal(MU, 0.6)
            @test priors.sigma == truncated(Normal(SIGMA, 0.6), eps(), Inf)
            @test priors.theta == Uniform(eps(), 5 * SIGMA)
        end

        @testset "background_priors" begin
            half_range = (WINDOW + BIN_SIZE) / 2  # DATA covers MU ± (WINDOW + BIN_SIZE) / 2

            # Without a peak: level over all bins
            priors = background_priors(DATA)
            level = _mean_background(DATA.weights, BIN_SIZE)
            lin_bound, quad_bound = _background_poly_bounds(level, half_range)
            @test keys(priors) == (:b, :b1, :b2, :h)
            @test priors.b == Uniform(0, 2 * level)
            @test priors.b1 == Uniform(-lin_bound, lin_bound)
            @test priors.b2 == Uniform(-quad_bound, quad_bound)
            @test priors.h == Uniform(0, 4 * level)

            # With a peak: level outside mu ± 3 sigma
            priors = background_priors(DATA; mu = MU, sigma = SIGMA)
            level = last(_get_peak_features(DATA; mu = MU, sigma = SIGMA))
            @test priors.b == Uniform(0, 2 * level)

            @test_throws ArgumentError background_priors(DATA; mu = MU)
        end

        @testset "Sanity floors yield valid prior bounds" begin
            EDGES = 0.5:1.0:5.5  # bin centers 1.0:5.0

            @testset "Zero background counts" begin
                data = Histogram(EDGES, [0, 0, 10, 0, 0])
                @test peak_priors(data; mu = 3.0, sigma = 0.5).n.b > 0
                # Mean background is floored at one count per bin:
                @test background_priors(data; mu = 3.0, sigma = 0.5).b.b ≈ 2.0
            end

            @testset "No peak above background" begin
                data = Histogram(EDGES, [4, 4, 0, 0, 4])
                @test peak_priors(data; mu = 3.0, sigma = 0.5).n.b > 0
                @test background_priors(data; mu = 3.0, sigma = 0.5).b.b > 0
            end
        end

    end

    @testset "build_prior from default blocks" begin

        @testset "Flat kernel takes the used entries" begin
            prior = build_prior(PEAK_KERNEL, merge(DEFAULTS.peak, DEFAULTS.bkg))
            @test propertynames(prior) == (:n, :alpha, :mu, :sigma, :theta, :b)
            @test prior.n == DEFAULTS.peak.n
            @test prior.b == DEFAULTS.bkg.b
        end

        @testset "Override via merge" begin
            defaults = merge(DEFAULTS.peak, DEFAULTS.bkg, (n = Uniform(0, A),))
            @test build_prior(PEAK_KERNEL, defaults).n == Uniform(0, A)
        end

        @testset "Nested kernel gives a nested prior" begin
            prior = build_prior(NESTED_KERNEL, DEFAULTS)
            @test propertynames(prior) == (:peak, :bkg)
            @test propertynames(prior.peak) == (:n, :mu, :sigma)
            @test propertynames(prior.bkg) == (:b,)
            @test prior.peak.n == DEFAULTS.peak.n
        end

        @testset "throws" begin
            # Missing entry
            @test_throws ArgumentError build_prior(NESTED_KERNEL, (peak = DEFAULTS.peak,))
            # Group entry is no NamedTuple
            @test_throws ArgumentError build_prior(
                NESTED_KERNEL,
                (peak = DEFAULTS.peak, bkg = Uniform(0, 1)),
            )
        end

    end

    @testset "build_posterior for @pf kernels" begin

        prior = build_prior(PEAK_KERNEL, merge(DEFAULTS.peak, DEFAULTS.bkg))

        @testset "Posterior of the kernel and the prior" begin
            posterior = build_posterior(DATA, PEAK_KERNEL, prior)
            @test posterior isa PosteriorMeasure
            @test isfinite(logdensityof(posterior, rand(prior)))
        end

        @testset "Throws when the prior does not match the kernel" begin
            prior_without_b = distprod(
                n = prior.n,
                alpha = prior.alpha,
                mu = prior.mu,
                sigma = prior.sigma,
                theta = prior.theta,
            )
            @test_throws ArgumentError build_posterior(DATA, PEAK_KERNEL, prior_without_b)
        end

        @testset "Nested kernel and prior" begin
            nested_prior = build_prior(NESTED_KERNEL, DEFAULTS)
            posterior = build_posterior(DATA, NESTED_KERNEL, nested_prior)
            @test isfinite(logdensityof(posterior, rand(nested_prior)))
        end

        @testset "Throws when a nested group does not match the kernel" begin
            nested_prior = build_prior(NESTED_KERNEL, DEFAULTS)
            peak_without_sigma =
                distprod(n = nested_prior.peak.n, mu = nested_prior.peak.mu)
            mismatched = distprod(peak = peak_without_sigma, bkg = nested_prior.bkg)
            @test_throws ArgumentError build_posterior(DATA, NESTED_KERNEL, mismatched)
        end

    end

end
