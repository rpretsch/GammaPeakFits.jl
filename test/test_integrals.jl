@testset "integrals" begin

    MU = 2048.0
    SIGMA = 10.0
    BIN_SIZE = 0.5
    A = 1000.0
    H = 50.0
    TAU = 5.0
    C_QUAD = 1.0
    C_LIN = 10.0
    C_CONST = 100.0
    WINDOW = 100.0

    bin_centers = collect((MU-WINDOW/2):BIN_SIZE:(MU+WINDOW/2))
    bin_edges = collect((MU-WINDOW/2-BIN_SIZE/2):BIN_SIZE:(MU+WINDOW/2+BIN_SIZE/2))
    data = SpectrumData(
        bin_centers = bin_centers,
        bin_edges = bin_edges,
        weights = ones(Int, length(bin_centers)),
        bin_size = BIN_SIZE,
    )

    @testset "Component integrals match numerical_integral" begin

        @testset "gaussian_integral" begin
            gaussian_params = GaussianParams(A = A, mu = MU, sigma = SIGMA)
            model_params = ModelParams(peak = PeakParams(gaussian = gaussian_params))
            integral = gaussian_integral(data, gaussian_params)
            @test length(integral) == length(bin_centers)
            @test isapprox(integral, numerical_integral(data, model_params))
        end

        @testset "compton_integral" begin
            compton_params = ComptonParams(h = H, mu = MU, sigma = SIGMA)
            model_params = ModelParams(peak = PeakParams(compton = compton_params))
            integral = compton_integral(data, compton_params)
            @test length(integral) == length(bin_centers)
            @test isapprox(integral, numerical_integral(data, model_params))
        end

        @testset "exGaussian_integral (low-energy tail)" begin
            params = ExGaussianParams(
                A = A,
                tau = TAU,
                is_lowEnergyTail = true,
                mu = MU,
                sigma = SIGMA,
            )
            model_params = ModelParams(peak = PeakParams(lowEnergyTail = params))
            integral = exGaussian_integral(data, params)
            @test length(integral) == length(bin_centers)
            @test isapprox(integral, numerical_integral(data, model_params))
        end

        @testset "exGaussian_integral (high-energy tail)" begin
            params = ExGaussianParams(
                A = A,
                tau = TAU,
                is_lowEnergyTail = false,
                mu = MU,
                sigma = SIGMA,
            )
            model_params = ModelParams(peak = PeakParams(highEnergyTail = params))
            integral = exGaussian_integral(data, params)
            @test length(integral) == length(bin_centers)
            @test isapprox(integral, numerical_integral(data, model_params))
        end

        @testset "quadPoly_integral" begin
            quad_params = QuadPolyParams(C = C_QUAD, mu = MU)
            model_params =
                ModelParams(background = BackgroundParams(quadPoly = quad_params))
            integral = quadPoly_integral(data, quad_params)
            @test length(integral) == length(bin_centers)
            @test isapprox(integral, numerical_integral(data, model_params))
        end

        @testset "linPoly_integral" begin
            lin_params = LinPolyParams(C = C_LIN, mu = MU)
            model_params = ModelParams(background = BackgroundParams(linPoly = lin_params))
            integral = linPoly_integral(data, lin_params)
            @test length(integral) == length(bin_centers)
            @test isapprox(integral, numerical_integral(data, model_params))
        end

        @testset "constPoly_integral" begin
            const_params = ConstPolyParams(C = C_CONST)
            model_params =
                ModelParams(background = BackgroundParams(constPoly = const_params))
            integral = constPoly_integral(data, const_params)
            @test length(integral) == length(bin_centers)
            @test isapprox(integral, numerical_integral(data, model_params))
        end

    end

    @testset "analytical_integral matches numerical_integral" begin

        model_params = ModelParams(
            peak = PeakParams(
                gaussian = GaussianParams(A = A, mu = MU, sigma = SIGMA),
                compton = ComptonParams(h = H, mu = MU, sigma = SIGMA),
                lowEnergyTail = ExGaussianParams(
                    A = A,
                    tau = TAU,
                    is_lowEnergyTail = true,
                    mu = MU,
                    sigma = SIGMA,
                ),
                highEnergyTail = ExGaussianParams(
                    A = A,
                    tau = TAU,
                    is_lowEnergyTail = false,
                    mu = MU,
                    sigma = SIGMA,
                ),
            ),
            background = BackgroundParams(
                quadPoly = QuadPolyParams(C = C_QUAD, mu = MU),
                linPoly = LinPolyParams(C = C_LIN, mu = MU),
                constPoly = ConstPolyParams(C = C_CONST),
            ),
        )

        analytical = analytical_integral(data, model_params)
        numerical = numerical_integral(data, model_params)

        @test length(analytical) == length(bin_centers)
        @test length(numerical) == length(bin_centers)
        @test isapprox(analytical, numerical)
        @test all(isfinite, analytical)
        @test all(isfinite, numerical)

    end

    @testset "analytical_integral skips disabled components" begin

        peak_only = ModelParams(
            peak = PeakParams(gaussian = GaussianParams(A = A, mu = MU, sigma = SIGMA)),
        )
        background_only = ModelParams(
            background = BackgroundParams(constPoly = ConstPolyParams(C = C_CONST)),
        )
        all_disabled = ModelParams(peak = PeakParams(), background = BackgroundParams())

        analytical_peak = analytical_integral(data, peak_only)
        analytical_background = analytical_integral(data, background_only)
        analytical_sum = analytical_peak .+ analytical_background
        analytical_both = analytical_integral(
            data,
            ModelParams(peak = peak_only.peak, background = background_only.background),
        )
        @test analytical_sum == analytical_both
        @test all(iszero, analytical_integral(data, all_disabled))
        @test analytical_peak ==
              gaussian_integral(data, GaussianParams(A = A, mu = MU, sigma = SIGMA))

    end

end
