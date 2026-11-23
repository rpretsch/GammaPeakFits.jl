using BAT: PosteriorMeasure, RandomWalk, TransformedMCMC, bat_sample, distprod, logdensityof
using Distributions: Normal, Uniform, truncated
using GammaPeakFits: mean_background, background_poly_bounds, get_peak_features
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
    CONFIGS = FitConfigs(mu = MU, sigma = SIGMA)
    PEAK_KERNEL = @pf $n * (
        (1 - $alpha) * GaussPeak($mu, $sigma) + $alpha * LowETailPeak($mu, $sigma, $theta)
    ) + PolynomialShape($b)
    BACKGROUND_KERNEL =
        @pf PolynomialShape($b, $b1, $b2; x0 = MU) + $h * GaussianStep(MU, SIGMA)

    @testset "default blocks" begin

        @testset "peak_priors" begin
            priors = peak_priors(DATA; mu = MU, sigma = SIGMA)
            _, peak_area, _ = get_peak_features(DATA; mu = MU, sigma = SIGMA)

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
            level = mean_background(DATA.weights, BIN_SIZE)
            lin_bound, quad_bound = background_poly_bounds(level, half_range)
            @test keys(priors) == (:b, :b1, :b2, :h)
            @test priors.b == Uniform(0, 2 * level)
            @test priors.b1 == Uniform(-lin_bound, lin_bound)
            @test priors.b2 == Uniform(-quad_bound, quad_bound)
            @test priors.h == Uniform(0, 4 * level)

            # With a peak: level outside mu ± 3 sigma
            priors = background_priors(DATA; mu = MU, sigma = SIGMA)
            level = last(get_peak_features(DATA; mu = MU, sigma = SIGMA))
            @test priors.b == Uniform(0, 2 * level)

            @test_throws ArgumentError background_priors(DATA; mu = MU)
        end

    end

    @testset "build_prior for @pf kernels" begin

        @testset "Names in kernel order with data-driven defaults" begin
            prior = build_prior(DATA, PEAK_KERNEL, CONFIGS)
            _, peak_area, background = get_peak_features(DATA; mu = MU, sigma = SIGMA)

            @test propertynames(prior) == (:n, :alpha, :mu, :sigma, :theta, :b)
            @test prior.n == Uniform(0, 3 * peak_area)
            @test prior.alpha == Uniform(0, 1)
            @test prior.mu == Normal(MU, 0.6)
            @test prior.sigma == truncated(Normal(SIGMA, 0.6), eps(), Inf)
            @test prior.theta == Uniform(eps(), 5 * SIGMA)
            @test prior.b == Uniform(0, 2 * background)
        end

        @testset "Background-only kernel estimates over the whole range" begin
            prior = build_prior(DATA, BACKGROUND_KERNEL, CONFIGS)
            background = mean_background(DATA.weights, BIN_SIZE)
            lin_bound, quad_bound = background_poly_bounds(background, WINDOW / 2)

            @test propertynames(prior) == (:b, :b1, :b2, :h)
            @test prior.b == Uniform(0, 2 * background)
            @test prior.b1 == Uniform(-lin_bound, lin_bound)
            @test prior.b2 == Uniform(-quad_bound, quad_bound)
            @test prior.h == Uniform(0, 4 * background)
        end

        @testset "Sanity floors yield valid prior bounds" begin
            configs = FitConfigs(mu = 3.0, sigma = 0.5)
            EDGES = 0.5:1.0:5.5  # bin centers 1.0:5.0

            @testset "Zero background counts" begin
                data = Histogram(EDGES, [0, 0, 10, 0, 0])
                prior = build_prior(data, PEAK_KERNEL, configs)
                @test prior.n.b > 0
                # Mean background is floored at one count per bin:
                @test prior.b.b ≈ 2.0
            end

            @testset "No peak above background" begin
                data = Histogram(EDGES, [4, 4, 0, 0, 4])
                prior = build_prior(data, PEAK_KERNEL, configs)
                @test prior.n.b > 0
                @test prior.b.b > 0
            end
        end

        @testset "configs.priors overrides a default" begin
            override = truncated(Normal(A, 10), 0, Inf)
            configs = FitConfigs(mu = MU, sigma = SIGMA, priors = [:n => override])
            prior = build_prior(DATA, PEAK_KERNEL, configs)
            @test prior.n == override
            @test prior.alpha == Uniform(0, 1)
        end

        @testset "throws" begin
            # Name without a default
            @test_throws ArgumentError build_prior(
                DATA,
                @pf($foo * GaussPeak(MU, SIGMA)),
                CONFIGS,
            )
            # Override for a name not in the kernel
            configs = FitConfigs(mu = MU, sigma = SIGMA, priors = [:h => Uniform(0, 1)])
            @test_throws ArgumentError build_prior(DATA, PEAK_KERNEL, configs)
            # Nested name
            @test_throws ArgumentError build_prior(
                DATA,
                @pf($p.n * GaussPeak(MU, SIGMA)),
                CONFIGS,
            )
        end

    end

    @testset "build_posterior for @pf kernels" begin

        prior = build_prior(DATA, PEAK_KERNEL, CONFIGS)

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

        @testset "End to end: kernel -> prior -> posterior -> bat_sample" begin
            posterior = build_posterior(DATA, PEAK_KERNEL, prior)
            samples = bat_sample(
                posterior,
                TransformedMCMC(proposal = RandomWalk(), nsteps = 1, nchains = 2),
            ).result
            @test all(isfinite, samples.logd)
            @test propertynames(first(samples).v) == propertynames(prior)
        end

    end

end
