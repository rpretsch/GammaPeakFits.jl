@testset "fitting" begin

    A = 1000.0
    MU = 2048.0
    SIGMA = 10.0
    PEAK_HEIGHT = 100.0
    PEAK_AREA = 1000.0
    DATA = SpectrumData(
        bin_centers = collect(2040.0:1.0:2050.0),
        weights = ones(Integer, 11),
        bin_size = 1.0,
    )
    C_CONST = 100.0
    C_LIN = 10.0

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
                MU,
                SIGMA;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )

        end

        @testset "Always includes mu prior" begin

            model_params = ModelParams(background = BackgroundParams(constPoly = true))
            prior = build_prior(
                model_params,
                MU,
                SIGMA;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            @test hasproperty(prior, :mu)
            @test prior.mu isa Normal
            @test prior.mu.μ == MU

        end

        @testset "sigma prior conditional on peak width components" begin

            model_params_noSigma = ModelParams(peak = PeakParams(compton = true))
            prior_noSigma = build_prior(
                model_params_noSigma,
                MU,
                SIGMA;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            @test !hasproperty(prior_noSigma, :sigma)

            model_params_sigma = ModelParams(peak = PeakParams(gaussian = true))

            prior_sigma = build_prior(
                model_params_sigma,
                MU,
                SIGMA;
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
                MU,
                SIGMA;
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
                MU,
                SIGMA;
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
                MU,
                SIGMA;
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
                    MU,
                    SIGMA;
                    peak_height = PEAK_HEIGHT,
                )
            end

            @testset "compton requires peak_height" begin
                model_params = ModelParams(peak = PeakParams(compton = true))
                @test_throws ArgumentError build_prior(
                    model_params,
                    MU,
                    SIGMA;
                    peak_area = PEAK_AREA,
                )
            end

            @testset "constPoly requires peak_height" begin
                model_params = ModelParams(background = BackgroundParams(constPoly = true))
                @test_throws ArgumentError build_prior(
                    model_params,
                    MU,
                    SIGMA;
                    peak_area = PEAK_AREA,
                )
            end

            @testset "lowEnergyTail requires peak_area" begin
                model_params = ModelParams(peak = PeakParams(lowEnergyTail = true))
                @test_throws ArgumentError build_prior(
                    model_params,
                    MU,
                    SIGMA;
                    peak_height = PEAK_HEIGHT,
                )
            end

            @testset "highEnergyTail requires peak_area" begin
                model_params = ModelParams(peak = PeakParams(highEnergyTail = true))
                @test_throws ArgumentError build_prior(
                    model_params,
                    MU,
                    SIGMA;
                    peak_height = PEAK_HEIGHT,
                )
            end

        end

        @testset "Disabled components produce no prior entries" begin

            model_params = ModelParams(peak = PeakParams(gaussian = true))
            prior = build_prior(
                model_params,
                MU,
                SIGMA;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            @test hasproperty(prior, :gaussian_A)
            @test !hasproperty(prior, :compton_h)
            @test !hasproperty(prior, :quadPoly_C)

        end

    end

    @testset "build_posterior" begin

        model_params = ModelParams(
            peak = PeakParams(gaussian = true),
            background = BackgroundParams(constPoly = true),
        )
        prior = build_prior(
            model_params,
            MU,
            SIGMA;
            peak_height = PEAK_HEIGHT,
            peak_area = PEAK_AREA,
        )

        @testset "Return type" begin

            posterior = build_posterior(DATA, prior)
            @test posterior isa PosteriorMeasure

        end

        @testset "Posterior with only peak (no background)" begin

            module_params_peak = ModelParams(peak = PeakParams(gaussian = true))
            prior_peak = build_prior(
                module_params_peak,
                MU,
                SIGMA;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            posterior = build_posterior(DATA, prior_peak)
            @test posterior isa PosteriorMeasure

        end

        @testset "Posterior with only background (no peak)" begin

            model_params_background =
                ModelParams(background = BackgroundParams(constPoly = true))
            prior_background = build_prior(
                model_params_background,
                MU,
                SIGMA;
                peak_height = PEAK_HEIGHT,
                peak_area = PEAK_AREA,
            )
            posterior = build_posterior(DATA, prior_background)
            @test posterior isa PosteriorMeasure

        end

    end

end
