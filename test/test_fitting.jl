using BAT:
    PosteriorMeasure, RandomWalk, TransformedMCMC, bat_sample, distprod, logdensityof
using Distributions: Normal, Poisson, Uniform, logpdf, truncated
using GammaPeakFits: mean_background, background_poly_bounds, get_peak_features
using StatsBase: Histogram

@testset "fitting" begin

    A = 1000.0
    MU = 2048.0
    SIGMA = 10.0
    WINDOW = 100.0
    BIN_SIZE = 1.0
    MODEL = ModelParams(
        peak = PeakParams(gaussian = GaussianParams(A = A, mu = MU, sigma = SIGMA)),
    )
    DATA = sample_spectrum(
        MODEL,
        BIN_SIZE;
        energy_range = (MU - WINDOW / 2 - BIN_SIZE / 2, MU + WINDOW / 2 + BIN_SIZE / 2),
    )
    SPECTRUM = SpectrumData(DATA)
    C_CONST = 100.0
    C_LIN = 10.0
    CONFIGS = FitConfigs(mu = MU, sigma = SIGMA, integration_method = Midpoint())
    PEAK_KERNEL = @pf $n * (
        (1 - $alpha) * GaussPeak($mu, $sigma) + $alpha * LowETailPeak($mu, $sigma, $theta)
    ) + PolynomialShape($b)
    BACKGROUND_KERNEL =
        @pf PolynomialShape($b, $b1, $b2; x0 = MU) + $h * GaussianStep(MU, SIGMA)

    @testset "poisson_ll" begin

        @testset "Basic likelihood with constant model" begin

            data = SpectrumData(
                bin_centers = [2048.0],
                bin_edges = [2047.5, 2048.5],
                weights = [10],
                bin_size = 1.0,
            )
            model_params = ModelParams(
                background = BackgroundParams(constPoly = ConstPolyParams(C = C_CONST)),
            )
            ll = poisson_ll(data, model_params, CONFIGS)
            expected_ll = logpdf(Poisson(C_CONST), 10)
            @test ll == expected_ll

        end

        @testset "Sums per-bin likelihoods exactly" begin

            data = SpectrumData(
                bin_centers = [2047.0, 2048.0, 2049.0],
                bin_edges = [2046.5, 2047.5, 2048.5, 2049.5],
                weights = [10, 14, 12],
                bin_size = 1.0,
            )
            model_params = ModelParams(
                background = BackgroundParams(constPoly = ConstPolyParams(C = C_CONST)),
            )
            ll = poisson_ll(data, model_params, CONFIGS)
            expected_ll = sum(logpdf.(Poisson(C_CONST), data.weights))
            @test ll == expected_ll

        end

        @testset "Likelihood with gaussian peak" begin

            model_params = ModelParams(
                peak = PeakParams(gaussian = GaussianParams(A = A, mu = MU, sigma = SIGMA)),
            )
            ll = poisson_ll(SPECTRUM, model_params, CONFIGS)
            @test isfinite(ll)

        end

        @testset "Reject negative model" begin

            model_params = ModelParams(
                background = BackgroundParams(linPoly = LinPolyParams(C = C_LIN, mu = MU)),
            )
            @test isinf(poisson_ll(SPECTRUM, model_params, CONFIGS))

        end

        @testset "Non-finite expected counts give -Inf" begin

            model_params = ModelParams(
                peak = PeakParams(
                    gaussian = GaussianParams(A = 1.0e308, mu = MU, sigma = 0.1),
                ),
            )
            ll = poisson_ll(
                SPECTRUM,
                model_params,
                FitConfigs(mu = MU, sigma = 0.1, integration_method = Midpoint()),
            )
            @test ll == -Inf

        end

        @testset "Integration method dispatch" begin

            model_params = ModelParams(
                background = BackgroundParams(constPoly = ConstPolyParams(C = C_CONST)),
            )

            ll_analytical = poisson_ll(
                SPECTRUM,
                model_params,
                FitConfigs(mu = MU, sigma = SIGMA, integration_method = Analytical()),
            )
            ll_numerical = poisson_ll(
                SPECTRUM,
                model_params,
                FitConfigs(mu = MU, sigma = SIGMA, integration_method = Numerical()),
            )
            ll_midpoint = poisson_ll(
                SPECTRUM,
                model_params,
                FitConfigs(mu = MU, sigma = SIGMA, integration_method = Midpoint()),
            )

            @test isfinite(ll_analytical)
            @test ll_analytical == ll_numerical
            @test ll_analytical == ll_midpoint

        end

    end

    @testset "build_prior" begin

        @testset "Throws on empty model" begin

            model_params = ModelParams()
            @test_throws ArgumentError build_prior(DATA, model_params, CONFIGS)

        end

        @testset "Throws on enabled-but-empty containers" begin

            model_params = ModelParams(peak = PeakParams())
            @test_throws ArgumentError build_prior(DATA, model_params, CONFIGS)

            model_params = ModelParams(background = BackgroundParams())
            @test_throws ArgumentError build_prior(DATA, model_params, CONFIGS)

            model_params = ModelParams(peak = PeakParams(), background = BackgroundParams())
            @test_throws ArgumentError build_prior(DATA, model_params, CONFIGS)

        end

        @testset "Throws on empty containers next to a valid block" begin

            model_params = ModelParams(
                peak = PeakParams(),
                background = BackgroundParams(constPoly = Enabled()),
            )
            @test_throws ArgumentError build_prior(DATA, model_params, CONFIGS)

            model_params = ModelParams(
                peak = PeakParams(gaussian = Enabled()),
                background = BackgroundParams(),
            )
            @test_throws ArgumentError build_prior(DATA, model_params, CONFIGS)

        end

        @testset "Throws on non-container peak and background values" begin

            model_params = ModelParams(peak = Enabled())
            @test_throws ArgumentError build_prior(DATA, model_params, CONFIGS)

            model_params = ModelParams(peak = GaussianParams(A = A, mu = MU, sigma = SIGMA))
            @test_throws ArgumentError build_prior(DATA, model_params, CONFIGS)

            model_params = ModelParams(background = Enabled())
            @test_throws ArgumentError build_prior(DATA, model_params, CONFIGS)

            model_params = ModelParams(background = ConstPolyParams(C = C_CONST))
            @test_throws ArgumentError build_prior(DATA, model_params, CONFIGS)

        end

        @testset "mu prior always except in constPoly background" begin

            model_params = ModelParams(background = BackgroundParams(constPoly = Enabled()))
            prior = build_prior(DATA, model_params, CONFIGS)
            @test !hasproperty(prior, :mu)
            model_params = ModelParams(peak = PeakParams(gaussian = Enabled()))
            prior = build_prior(DATA, model_params, CONFIGS)
            @test hasproperty(prior, :mu)
            @test prior.mu isa Normal
            @test prior.mu.μ == MU

        end

        @testset "sigma prior conditional on peak components" begin

            model_params_noSigma =
                ModelParams(background = BackgroundParams(quadPoly = Enabled()))
            prior_noSigma = build_prior(DATA, model_params_noSigma, CONFIGS)
            @test !hasproperty(prior_noSigma, :sigma)

            model_params_sigma = ModelParams(peak = PeakParams(gaussian = Enabled()))

            prior_sigma = build_prior(DATA, model_params_sigma, CONFIGS)
            @test hasproperty(prior_sigma, :sigma)
            @test prior_sigma.sigma.untruncated isa Normal
            @test prior_sigma.sigma.untruncated.μ == SIGMA

        end

        @testset "Disabled components produce no prior entries" begin

            model_params = ModelParams(peak = PeakParams(gaussian = Enabled()))
            prior = build_prior(DATA, model_params, CONFIGS)
            @test hasproperty(prior, :gaussian_A)
            @test !hasproperty(prior, :compton_h)
            @test !hasproperty(prior, :quadPoly_C)

        end

        @testset "Polynomial background priors scale with the data" begin

            model_params = ModelParams(
                background = BackgroundParams(quadPoly = Enabled(), linPoly = Enabled()),
            )
            prior = build_prior(DATA, model_params, CONFIGS)

            # Background-only model: the level is estimated over the whole data range
            background = mean_background(DATA.weights, BIN_SIZE)
            lin_bound, quad_bound = background_poly_bounds(background, BIN_SIZE, CONFIGS)

            @test prior.linPoly_C isa Uniform
            @test prior.linPoly_C.a == - lin_bound
            @test prior.linPoly_C.b == lin_bound
            @test prior.quadPoly_C isa Uniform
            @test prior.quadPoly_C.a == - quad_bound
            @test prior.quadPoly_C.b == quad_bound

        end

        @testset "Tail priors scale with the core width" begin

            model_params = ModelParams(
                peak = PeakParams(
                    gaussian = Enabled(),
                    lowEnergyTail = Enabled(),
                    highEnergyTail = Enabled(),
                ),
            )
            prior = build_prior(DATA, model_params, CONFIGS)

            @test prior.lowEnergyTail_tau isa Uniform
            @test prior.lowEnergyTail_tau.a == eps()
            @test prior.lowEnergyTail_tau.b == 5 * SIGMA
            @test prior.highEnergyTail_tau isa Uniform
            @test prior.highEnergyTail_tau.a == eps()
            @test prior.highEnergyTail_tau.b == 5 * SIGMA

        end

        @testset "Sanity floors yield valid prior bounds" begin

            model_params = ModelParams(
                peak = PeakParams(gaussian = Enabled()),
                background = BackgroundParams(constPoly = Enabled()),
            )
            configs = FitConfigs(mu = 3.0, sigma = 0.5)
            EDGES = 0.5:1.0:5.5  # bin centers 1.0:5.0

            @testset "Zero background counts" begin

                data = Histogram(EDGES, [0, 0, 10, 0, 0])
                prior = build_prior(data, model_params, configs)

                @test prior.gaussian_A isa Uniform
                @test prior.constPoly_C isa Uniform
                @test prior.gaussian_A.b > 0
                # Mean background is floored at one count per bin:
                @test prior.constPoly_C.b ≈ 2.0

            end

            @testset "No peak above background" begin

                data = Histogram(EDGES, [4, 4, 0, 0, 4])
                prior = build_prior(data, model_params, configs)

                @test prior.gaussian_A isa Uniform
                @test prior.gaussian_A.b > 0
                @test prior.constPoly_C isa Uniform
                @test prior.constPoly_C.b > 0

            end

        end

        @testset "Custom priors propagate into the combined prior" begin

            configs = FitConfigs(
                mu = MU,
                sigma = SIGMA,
                priors = [:gaussian_A => truncated(Normal(1000, 10), 0, Inf)],
            )
            model_params = ModelParams(peak = PeakParams(gaussian = Enabled()))
            combined_prior = build_prior(DATA, model_params, configs)

            @test combined_prior.gaussian_A.untruncated isa Normal
            @test combined_prior.gaussian_A.untruncated.μ == 1000
            @test combined_prior.gaussian_A.untruncated.σ == 10

        end

        @testset "Unknown custom priors are rejected" begin

            configs = FitConfigs(
                mu = MU,
                sigma = SIGMA,
                priors = [:gaussian_smth => truncated(Normal(1000, 10), 0, Inf)],
            )
            model_params = ModelParams(peak = PeakParams(gaussian = Enabled()))
            @test_throws ArgumentError build_prior(DATA, model_params, configs)

        end

    end

    @testset "build_prior for @pf kernels" begin

        @testset "Names in kernel order with data-driven defaults" begin
            prior = build_prior(DATA, PEAK_KERNEL, CONFIGS)
            _, peak_area, background = get_peak_features(DATA, CONFIGS)

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
            lin_bound, quad_bound = background_poly_bounds(background, BIN_SIZE, CONFIGS)

            @test propertynames(prior) == (:b, :b1, :b2, :h)
            @test prior.b == Uniform(0, 2 * background)
            @test prior.b1 == Uniform(-lin_bound, lin_bound)
            @test prior.b2 == Uniform(-quad_bound, quad_bound)
            @test prior.h == Uniform(0, 4 * background)
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

    @testset "build_posterior" begin

        model_params = ModelParams(
            peak = PeakParams(gaussian = Enabled()),
            background = BackgroundParams(constPoly = Enabled()),
        )
        prior = build_prior(DATA, model_params, CONFIGS)

        @testset "Return type" begin

            posterior = build_posterior(DATA, prior, CONFIGS)
            @test posterior isa PosteriorMeasure

            param_values = (mu = MU, sigma = SIGMA, gaussian_A = A, constPoly_C = C_CONST)
            expected = poisson_ll(
                SPECTRUM,
                ModelParams(
                    peak = PeakParams(
                        gaussian = GaussianParams(A = A, mu = MU, sigma = SIGMA),
                    ),
                    background = BackgroundParams(constPoly = ConstPolyParams(C = C_CONST)),
                ),
                CONFIGS,
            )
            @test posterior.likelihood._log_f(param_values) == expected
            @test isfinite(expected)

        end

        @testset "Posterior with only peak (no background)" begin

            model_params_peak = ModelParams(peak = PeakParams(gaussian = Enabled()))
            prior_peak = build_prior(DATA, model_params_peak, CONFIGS)
            posterior = build_posterior(DATA, prior_peak, CONFIGS)
            @test posterior isa PosteriorMeasure

            param_values = (mu = MU, sigma = SIGMA, gaussian_A = A)
            expected = poisson_ll(
                SPECTRUM,
                ModelParams(
                    peak = PeakParams(
                        gaussian = GaussianParams(A = A, mu = MU, sigma = SIGMA),
                    ),
                ),
                CONFIGS,
            )
            @test posterior.likelihood._log_f(param_values) == expected
            @test isfinite(expected)

        end

        @testset "Posterior with only background (no peak)" begin

            model_params_background =
                ModelParams(background = BackgroundParams(constPoly = Enabled()))
            prior_background = build_prior(DATA, model_params_background, CONFIGS)
            posterior = build_posterior(DATA, prior_background, CONFIGS)
            @test posterior isa PosteriorMeasure

            param_values = (constPoly_C = C_CONST,)
            expected = poisson_ll(
                SPECTRUM,
                ModelParams(
                    background = BackgroundParams(constPoly = ConstPolyParams(C = C_CONST)),
                ),
                CONFIGS,
            )
            @test posterior.likelihood._log_f(param_values) == expected
            @test isfinite(expected)

        end

        @testset "Unphysical parameters give -Inf" begin

            model_params = ModelParams(background = BackgroundParams(linPoly = Enabled()))
            prior = build_prior(DATA, model_params, CONFIGS)
            posterior = build_posterior(DATA, prior, CONFIGS)

            param_values = (mu = MU, linPoly_C = C_LIN)
            @test posterior.likelihood._log_f(param_values) == -Inf

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
            samples =
                bat_sample(
                    posterior,
                    TransformedMCMC(proposal = RandomWalk(), nsteps = 1, nchains = 2),
                ).result
            @test all(isfinite, samples.logd)
            @test propertynames(first(samples).v) == propertynames(prior)
        end

    end

end
