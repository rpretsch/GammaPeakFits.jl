using BAT: PosteriorMeasure
using Distributions: Normal, Poisson, logpdf, truncated

@testset "fitting" begin

    A = 1000.0
    MU = 2048.0
    SIGMA = 10.0
    WINDOW = 100.0
    model = ModelParams(
        peak = PeakParams(gaussian = GaussianParams(A = A, mu = MU, sigma = SIGMA)),
    )
    DATA = SpectrumData((MU-WINDOW/2), (MU+WINDOW/2), 1.0, model)
    C_CONST = 100.0
    C_LIN = 10.0
    CONFIGS = FitConfigs(mu = MU, sigma = SIGMA, integration_method = Midpoint())

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
            ll = poisson_ll(DATA, model_params, CONFIGS)
            @test isfinite(ll)

        end

        @testset "Reject negative model" begin

            model_params = ModelParams(
                background = BackgroundParams(linPoly = LinPolyParams(C = C_LIN, mu = MU)),
            )
            @test isinf(poisson_ll(DATA, model_params, CONFIGS))

        end

        @testset "Non-finite expected counts give -Inf" begin

            model_params = ModelParams(
                peak = PeakParams(
                    gaussian = GaussianParams(A = 1.0e308, mu = MU, sigma = 0.1),
                ),
            )
            ll = poisson_ll(
                DATA,
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
                DATA,
                model_params,
                FitConfigs(mu = MU, sigma = SIGMA, integration_method = Analytical()),
            )
            ll_numerical = poisson_ll(
                DATA,
                model_params,
                FitConfigs(mu = MU, sigma = SIGMA, integration_method = Numerical()),
            )
            ll_midpoint = poisson_ll(
                DATA,
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

        @testset "Custom priors propagate into the combined prior" begin

            priors = PriorPair[:gaussian_A=>truncated(Normal(1000, 10), 0, Inf)]
            configs = FitConfigs(mu = MU, sigma = SIGMA)
            model_params = ModelParams(peak = PeakParams(gaussian = Enabled()))
            combined_prior = build_prior(DATA, model_params, configs; priors = priors)

            @test combined_prior.gaussian_A.untruncated isa Normal
            @test combined_prior.gaussian_A.untruncated.μ == 1000
            @test combined_prior.gaussian_A.untruncated.σ == 10

        end

        @testset "Unknown custom priors are regected" begin

            priors = PriorPair[:gaussian_smth=>truncated(Normal(1000, 10), 0, Inf)]
            configs = FitConfigs(mu = MU, sigma = SIGMA)
            model_params = ModelParams(peak = PeakParams(gaussian = Enabled()))
            @test_throws ArgumentError build_prior(
                DATA,
                model_params,
                configs;
                priors = priors,
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

            v = (mu = MU, sigma = SIGMA, gaussian_A = A, constPoly_C = C_CONST)
            expected = poisson_ll(
                DATA,
                ModelParams(
                    peak = PeakParams(
                        gaussian = GaussianParams(A = A, mu = MU, sigma = SIGMA),
                    ),
                    background = BackgroundParams(constPoly = ConstPolyParams(C = C_CONST)),
                ),
                CONFIGS,
            )
            @test posterior.likelihood._log_f(v) == expected
            @test isfinite(expected)

        end

        @testset "Posterior with only peak (no background)" begin

            module_params_peak = ModelParams(peak = PeakParams(gaussian = Enabled()))
            prior_peak = build_prior(DATA, model_params, CONFIGS)
            posterior = build_posterior(DATA, prior_peak, CONFIGS)
            @test posterior isa PosteriorMeasure

            v = (mu = MU, sigma = SIGMA, gaussian_A = A)
            expected = poisson_ll(
                DATA,
                ModelParams(
                    peak = PeakParams(
                        gaussian = GaussianParams(A = A, mu = MU, sigma = SIGMA),
                    ),
                ),
                CONFIGS,
            )
            @test posterior.likelihood._log_f(v) == expected
            @test isfinite(expected)

        end

        @testset "Posterior with only background (no peak)" begin

            model_params_background =
                ModelParams(background = BackgroundParams(constPoly = Enabled()))
            prior_background = build_prior(DATA, model_params, CONFIGS)
            posterior = build_posterior(DATA, prior_background, CONFIGS)
            @test posterior isa PosteriorMeasure

            v = (constPoly_C = C_CONST,)
            expected = poisson_ll(
                DATA,
                ModelParams(
                    background = BackgroundParams(constPoly = ConstPolyParams(C = C_CONST)),
                ),
                CONFIGS,
            )
            @test posterior.likelihood._log_f(v) == expected
            @test isfinite(expected)

        end

        @testset "Unphysical parameters give -Inf" begin

            model_params = ModelParams(background = BackgroundParams(linPoly = Enabled()))
            prior = build_prior(DATA, model_params, CONFIGS)
            posterior = build_posterior(DATA, prior, CONFIGS)

            v = (mu = MU, linPoly_C = C_LIN)
            @test posterior.likelihood._log_f(v) == -Inf

        end

    end

end
