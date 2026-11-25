using BAT: distprod, logdensityof
using Distributions: Normal, Uniform, truncated
using GammaPeakFits:
    _mean_background, _background_poly_bounds, _background_mask, _get_peak_features
using StatsBase: Histogram, midpoints

@testset "fitting" begin

    A = 1000.0
    MU = 2048.0
    SIGMA = 10.0
    WINDOW = 200.0
    BIN_SIZE = 1.0

    DATA = sample_spectrum(
        A * GaussPeak(MU, SIGMA);
        bin_size = BIN_SIZE,
        energy_range = (MU - WINDOW / 2 - BIN_SIZE / 2, MU + WINDOW / 2 + BIN_SIZE / 2),
    )

    PEAK_KERNEL = @pf $n * (
        (1 - $alpha) * GaussPeak($mu, $sigma) + $alpha * LowETailPeak($mu, $sigma, $theta)
    ) + PolynomialShape($b0)
    NESTED_KERNEL =
        @pf $peak.n * GaussPeak($peak.mu, $peak.sigma) + PolynomialShape($bkg.b0)

    DEFAULTS = (
        peak = peak_priors(DATA; mu = MU, sigma = SIGMA),
        bkg = background_priors(DATA; mu = MU, sigma = SIGMA),
    )

    UNEVEN_HIST = Histogram([0.0, 1.0, 3.0], [1, 2])

    @testset "default blocks" begin

        @testset "peak_priors" begin

            @testset "one peak" begin
                priors = peak_priors(DATA; mu = MU, sigma = SIGMA)
                _, peak_area, _ = _get_peak_features(DATA; mu = MU, sigma = SIGMA)

                @test keys(priors) == (:n, :alpha, :mu, :sigma, :theta)
                @test priors.n == Uniform(0, 3 * peak_area)
                @test priors.alpha == Uniform(0, 1)
                @test priors.mu == Normal(MU, 0.6)
                @test priors.sigma == truncated(Normal(SIGMA, 0.6), eps(), Inf)
                @test priors.theta == Uniform(eps(), 5 * SIGMA)
            end

            @testset "selected one of several peaks" begin
                mus = [MU - 40, MU]
                sigma = 2.0
                priors = peak_priors(DATA; mu = mus, sigma, peak_index = 2)
                _, peak_area, _ = _get_peak_features(DATA; mu = mus, sigma, peak_index = 2)

                @test priors.n == Uniform(0, 3 * peak_area)
                @test priors.mu == Normal(MU, 0.6)
                @test priors.sigma == truncated(Normal(sigma, 0.6), eps(), Inf)
                @test priors.theta == Uniform(eps(), 5 * sigma)
            end

            @testset "throws" begin
                # sigma not positive, rejected by _get_peak_features
                @test_throws ArgumentError peak_priors(DATA; mu = MU, sigma = 0.0)
            end

        end

        @testset "background_priors" begin

            @testset "without a peak, level over all bins" begin
                # DATA covers MU ± (WINDOW + BIN_SIZE) / 2
                half_range = (WINDOW + BIN_SIZE) / 2
                priors = background_priors(DATA)
                level = _mean_background(DATA.weights, BIN_SIZE)
                lin_bound, quad_bound = _background_poly_bounds(level, half_range)
                @test keys(priors) == (:b0, :b1, :b2, :h)
                @test priors.b0 == Uniform(0, 2 * level)
                @test priors.b1 == Uniform(-lin_bound, lin_bound)
                @test priors.b2 == Uniform(-quad_bound, quad_bound)
                @test priors.h == Uniform(0, 4 * level)
            end

            @testset "with peaks, level outside mu ± 5 sigma of all of them" begin
                mus = [MU, MU + 60]
                priors = background_priors(DATA; mu = mus, sigma = SIGMA)
                centers = midpoints(only(DATA.edges))
                mask = _background_mask(centers; mu = mus, sigma = [SIGMA, SIGMA])
                level = _mean_background(DATA.weights[mask], BIN_SIZE)
                @test priors.b0 == Uniform(0, 2 * level)
            end

            @testset "throws" begin
                # Only one of mu and sigma given
                @test_throws ArgumentError background_priors(DATA; mu = MU)
                # Uneven bins without a peak, rejected by _check_spectrum
                @test_throws ArgumentError background_priors(UNEVEN_HIST)
                # Two widths for one peak, rejected by _peak_params
                @test_throws ArgumentError background_priors(
                    DATA;
                    mu = MU,
                    sigma = [SIGMA, SIGMA],
                )
                # mu ± 5 sigma covers all bins, rejected by _background_mask
                @test_throws ArgumentError background_priors(DATA; mu = MU, sigma = 30.0)
            end

        end

    end

    @testset "build_prior" begin

        @testset "flat kernel takes the used entries" begin
            prior = build_prior(PEAK_KERNEL, merge(DEFAULTS.peak, DEFAULTS.bkg))
            @test propertynames(prior) == (:n, :alpha, :mu, :sigma, :theta, :b0)
            @test prior.n == DEFAULTS.peak.n
            @test prior.b0 == DEFAULTS.bkg.b0
        end

        @testset "override via merge" begin
            defaults = merge(DEFAULTS.peak, DEFAULTS.bkg, (n = Uniform(0, A),))
            @test build_prior(PEAK_KERNEL, defaults).n == Uniform(0, A)
        end

        @testset "nested kernel gives a nested prior" begin
            prior = build_prior(NESTED_KERNEL, DEFAULTS)
            @test propertynames(prior) == (:peak, :bkg)
            @test propertynames(prior.peak) == (:n, :mu, :sigma)
            @test propertynames(prior.bkg) == (:b0,)
            @test prior.peak.n == DEFAULTS.peak.n
        end

        @testset "kernel using a group as a whole takes its entry" begin
            whole_group_kernel =
                @pf getproperty($peak, :n) * GaussPeak(getproperty($peak, :mu), SIGMA)
            prior = build_prior(whole_group_kernel, DEFAULTS)
            @test propertynames(prior) == (:peak,)
            @test propertynames(prior.peak) == keys(DEFAULTS.peak)
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

    @testset "build_posterior" begin

        prior = build_prior(PEAK_KERNEL, merge(DEFAULTS.peak, DEFAULTS.bkg))

        @testset "posterior of the kernel and the prior" begin
            posterior = build_posterior(DATA, PEAK_KERNEL, prior)
            @test isfinite(logdensityof(posterior, rand(prior)))
        end

        @testset "nested kernel and prior" begin
            nested_prior = build_prior(NESTED_KERNEL, DEFAULTS)
            posterior = build_posterior(DATA, NESTED_KERNEL, nested_prior)
            @test isfinite(logdensityof(posterior, rand(nested_prior)))
        end

        @testset "throws" begin
            # Uneven bins, rejected by _check_spectrum
            @test_throws ArgumentError build_posterior(UNEVEN_HIST, PEAK_KERNEL, prior)
            # Prior without `b0` does not match the kernel
            prior_without_b0 = distprod(
                n = prior.n,
                alpha = prior.alpha,
                mu = prior.mu,
                sigma = prior.sigma,
                theta = prior.theta,
            )
            @test_throws ArgumentError build_posterior(DATA, PEAK_KERNEL, prior_without_b0)
            # Nested group `peak` without `sigma` does not match the kernel
            nested_prior = build_prior(NESTED_KERNEL, DEFAULTS)
            peak_without_sigma =
                distprod(n = nested_prior.peak.n, mu = nested_prior.peak.mu)
            mismatched = distprod(peak = peak_without_sigma, bkg = nested_prior.bkg)
            @test_throws ArgumentError build_posterior(DATA, NESTED_KERNEL, mismatched)
            # Group `peak` of the prior is not nested, but the kernel uses `peak.n`, …
            flat_group = distprod(peak = Uniform(0, 1), bkg = nested_prior.bkg)
            @test_throws ArgumentError build_posterior(DATA, NESTED_KERNEL, flat_group)
        end

    end

end
