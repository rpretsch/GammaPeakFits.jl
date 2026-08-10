@testset "fitting" begin

    A = 1000.0
    MU = 2048.0
    SIGMA = 10.0
    PEAK_HEIGHT = 100.0
    PEAK_AREA = 1000.0
    DATA = SpectrumData(
        bin_centers = collect(2040.0:1.0:2050.0),
        weights = ones(Int, 11),
        bin_size = 1.0,
    )
    C_CONST = 100.0
    C_LIN = 10.0
    CONFIGS = Configs(mu = MU, sigma = SIGMA)

    @testset "poisson_ll" begin

        @testset "Basic likelihood with constant model" begin

            data = SpectrumData(bin_centers = [2048.0], weights = [10], bin_size = 1.0)
            model_params = ModelParams(
                background = BackgroundParams(constPoly = ConstPolyParams(C = C_CONST)),
            )
            ll = poisson_ll(data, model_params)
            expected_ll = logpdf(Poisson(C_CONST), 10)
            @test ll == expected_ll

        end

        @testset "Multi-bin likelihood sums correctly" begin

            model_params = ModelParams(
                background = BackgroundParams(constPoly = ConstPolyParams(C = C_CONST)),
            )
            ll = poisson_ll(DATA, model_params)
            @test ll isa AbstractFloat
            @test !isnan(ll)
            @test !isinf(ll)

        end

        @testset "Likelihood with gaussian peak" begin

            model_params = ModelParams(
                peak = PeakParams(gaussian = GaussianParams(A = A, mu = MU, sigma = SIGMA)),
            )
            ll = poisson_ll(DATA, model_params)
            @test ll isa AbstractFloat
            @test !isnan(ll)

        end

        @testset "Reject negative model" begin

            model_params = ModelParams(
                background = BackgroundParams(linPoly = LinPolyParams(C = C_LIN, mu = MU)),
            )
            @test isinf(poisson_ll(DATA, model_params))

        end

    end

    @testset "build_prior" begin

        @testset "Throws on empty model" begin

            model_params = ModelParams()
            @test_throws ArgumentError build_prior(
                model_params,
                CONFIGS;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )

        end

        @testset "mu prior always except in constPoly background" begin

            model_params = ModelParams(background = BackgroundParams(constPoly = true))
            prior = build_prior(
                model_params,
                CONFIGS;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            @test !hasproperty(prior, :mu)
            model_params = ModelParams(peak = PeakParams(gaussian = true))
            prior = build_prior(
                model_params,
                CONFIGS;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            @test hasproperty(prior, :mu)
            @test prior.mu isa Normal
            @test prior.mu.μ == MU

        end

        @testset "sigma prior conditional on peak components" begin

            model_params_noSigma =
                ModelParams(background = BackgroundParams(quadPoly = true))
            prior_noSigma = build_prior(
                model_params_noSigma,
                CONFIGS;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            @test !hasproperty(prior_noSigma, :sigma)

            model_params_sigma = ModelParams(peak = PeakParams(gaussian = true))

            prior_sigma = build_prior(
                model_params_sigma,
                CONFIGS;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            @test hasproperty(prior_sigma, :sigma)
            @test prior_sigma.sigma.untruncated isa Normal
            @test prior_sigma.sigma.untruncated.μ == SIGMA

        end

        @testset "Prior bounds use the passed parameters" begin

            model_params = ModelParams(peak = PeakParams(gaussian = true, compton = true))
            prior = build_prior(
                model_params,
                CONFIGS;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            @test prior.gaussian_A isa Uniform
            @test prior.gaussian_A.b == PEAK_AREA
            @test prior.compton_h isa Uniform
            @test prior.compton_h.b == PEAK_HEIGHT

            model_params = ModelParams(background = BackgroundParams(constPoly = true))
            prior = build_prior(
                model_params,
                CONFIGS;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            @test prior.constPoly_C isa Uniform
            @test prior.constPoly_C.b == PEAK_HEIGHT

        end

        @testset "Return type" begin

            model_params = ModelParams(peak = PeakParams(gaussian = true))
            prior = build_prior(
                model_params,
                CONFIGS;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            @test prior isa NamedTupleDist

        end

        @testset "Throws on missing parameters" begin

            @testset "gaussian requires peak_area" begin
                model_params = ModelParams(peak = PeakParams(gaussian = true))
                @test_throws ArgumentError build_prior(
                    model_params,
                    CONFIGS;
                    peak_height = PEAK_HEIGHT,
                )
            end

            @testset "compton requires peak_height" begin
                model_params = ModelParams(peak = PeakParams(compton = true))
                @test_throws ArgumentError build_prior(
                    model_params,
                    CONFIGS;
                    peak_area = PEAK_AREA,
                )
            end

            @testset "constPoly requires peak_height" begin
                model_params = ModelParams(background = BackgroundParams(constPoly = true))
                @test_throws ArgumentError build_prior(
                    model_params,
                    CONFIGS;
                    peak_area = PEAK_AREA,
                )
            end

            @testset "lowEnergyTail requires peak_area" begin
                model_params = ModelParams(peak = PeakParams(lowEnergyTail = true))
                @test_throws ArgumentError build_prior(
                    model_params,
                    CONFIGS;
                    peak_height = PEAK_HEIGHT,
                )
            end

            @testset "highEnergyTail requires peak_area" begin
                model_params = ModelParams(peak = PeakParams(highEnergyTail = true))
                @test_throws ArgumentError build_prior(
                    model_params,
                    CONFIGS;
                    peak_height = PEAK_HEIGHT,
                )
            end

        end

        @testset "Disabled components produce no prior entries" begin

            model_params = ModelParams(peak = PeakParams(gaussian = true))
            prior = build_prior(
                model_params,
                CONFIGS;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            @test hasproperty(prior, :gaussian_A)
            @test !hasproperty(prior, :compton_h)
            @test !hasproperty(prior, :quadPoly_C)

        end

        @testset "Configs values propagate into the priors" begin

            configs = Configs(
                mu = MU,
                sigma = SIGMA,
                mu_std = 1.0,
                sigma_std = 2.0,
                lowEnergyTail_tau_upper = 5.0,
                highEnergyTail_tau_upper = 6.0,
                quadPoly_C_limits = (-2.0, 2.0),
                linPoly_C_limits = (-3.0, 3.0),
            )
            model_params = ModelParams(
                peak = PeakParams(
                    gaussian = true,
                    lowEnergyTail = true,
                    highEnergyTail = true,
                ),
                background = BackgroundParams(quadPoly = true, linPoly = true),
            )
            prior = build_prior(
                model_params,
                configs;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )

            @test prior.mu isa Normal
            @test prior.mu.μ == MU
            @test prior.mu.σ == 1.0
            @test prior.sigma.untruncated isa Normal
            @test prior.sigma.untruncated.μ == SIGMA
            @test prior.sigma.untruncated.σ == 2.0
            @test prior.lowEnergyTail_tau isa Uniform
            @test prior.lowEnergyTail_tau.a == eps()
            @test prior.lowEnergyTail_tau.b == 5.0
            @test prior.highEnergyTail_tau isa Uniform
            @test prior.highEnergyTail_tau.b == 6.0
            @test prior.quadPoly_C isa Uniform
            @test prior.quadPoly_C.a == -2.0
            @test prior.quadPoly_C.b == 2.0
            @test prior.linPoly_C isa Uniform
            @test prior.linPoly_C.a == -3.0
            @test prior.linPoly_C.b == 3.0

        end

    end

    @testset "build_posterior" begin

        model_params = ModelParams(
            peak = PeakParams(gaussian = true),
            background = BackgroundParams(constPoly = true),
        )
        prior = build_prior(
            model_params,
            CONFIGS;
            peak_height = PEAK_HEIGHT,
            peak_area = PEAK_AREA,
        )

        @testset "Return type" begin

            posterior = build_posterior(DATA, prior)
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
            )
            @test posterior.likelihood._log_f(v) == expected
            @test isfinite(expected)

        end

        @testset "Posterior with only peak (no background)" begin

            module_params_peak = ModelParams(peak = PeakParams(gaussian = true))
            prior_peak = build_prior(
                module_params_peak,
                CONFIGS;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            posterior = build_posterior(DATA, prior_peak)
            @test posterior isa PosteriorMeasure

            v = (mu = MU, sigma = SIGMA, gaussian_A = A)
            expected = poisson_ll(
                DATA,
                ModelParams(
                    peak = PeakParams(
                        gaussian = GaussianParams(A = A, mu = MU, sigma = SIGMA),
                    ),
                ),
            )
            @test posterior.likelihood._log_f(v) == expected
            @test isfinite(expected)

        end

        @testset "Posterior with only background (no peak)" begin

            model_params_background =
                ModelParams(background = BackgroundParams(constPoly = true))
            prior_background = build_prior(
                model_params_background,
                CONFIGS;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            posterior = build_posterior(DATA, prior_background)
            @test posterior isa PosteriorMeasure

            v = (mu = MU, constPoly_C = C_CONST)
            expected = poisson_ll(
                DATA,
                ModelParams(
                    background = BackgroundParams(constPoly = ConstPolyParams(C = C_CONST)),
                ),
            )
            @test posterior.likelihood._log_f(v) == expected
            @test isfinite(expected)

        end

        @testset "Unphysical parameters give -Inf" begin

            model_params = ModelParams(background = BackgroundParams(linPoly = true))
            prior = build_prior(
                model_params,
                CONFIGS;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            posterior = build_posterior(DATA, prior)

            v = (mu = MU, linPoly_C = C_LIN)
            @test posterior.likelihood._log_f(v) == -Inf

        end

    end

end
