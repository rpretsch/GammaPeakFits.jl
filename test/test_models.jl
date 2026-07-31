@testset "models" begin

    MU = 2048.0
    SIGMA = 10.0
    X_ARRAY = [MU-3*SIGMA, MU-2*SIGMA, MU-SIGMA, MU, MU+SIGMA, MU+2*SIGMA, MU+3*SIGMA]
    A = 1000.0
    H = 50.0
    LAMBDA = 0.1
    C_QUAD = 1.0
    C_LIN = 10.0
    C_CONST = 100.0

    @testset "Component functions" begin

        @testset "gaussian" begin

            gaussian_params = GaussianParams(A = A, mu = MU, sigma = SIGMA)
            result = gaussian(X_ARRAY, gaussian_params)
            @test result == gaussian.(X_ARRAY, Ref(gaussian_params))

            expected = @. A * pdf(Normal(MU, SIGMA), X_ARRAY)
            @test result == expected

        end

        @testset "compton" begin

            compton_params = ComptonParams(h = H, mu = MU, sigma = SIGMA)
            result = compton(X_ARRAY, compton_params)
            @test result == compton.(X_ARRAY, Ref(compton_params))

            expected = @. H/2 * erfc((X_ARRAY - MU) / (sqrt(2) * SIGMA))
            @test result == expected

        end

        @testset "exGaussian" begin

            exGaussian_params =
                ExGaussianParams(A = A, lambda = LAMBDA, mu = MU, sigma = SIGMA)
            result = exGaussian(X_ARRAY, exGaussian_params)
            @test result == exGaussian.(X_ARRAY, Ref(exGaussian_params))

            expected = @. A * LAMBDA/2 *
               exp(LAMBDA/2 * (2 * MU + LAMBDA * SIGMA^2 - 2 * X_ARRAY)) *
               erfc((MU + LAMBDA * SIGMA^2 - X_ARRAY)/(sqrt(2) * SIGMA))
            @test result == expected

        end

        @testset "quad_polynomial" begin

            quadPoly_params = QuadPolyParams(C = C_QUAD, mu = MU)
            result = quad_polynomial(X_ARRAY, quadPoly_params)
            @test result == quad_polynomial.(X_ARRAY, Ref(quadPoly_params))

            expected = @. C_QUAD * (X_ARRAY - MU)^2
            @test result == expected

        end

        @testset "lin_polynomial" begin

            linPoly_params = LinPolyParams(C = C_LIN, mu = MU)
            result = lin_polynomial(X_ARRAY, linPoly_params)
            @test result == lin_polynomial.(X_ARRAY, Ref(linPoly_params))

            expected = @. C_LIN * (X_ARRAY - MU)
            @test result == expected

        end

        @testset "const_polynomial" begin

            constPoly_params = ConstPolyParams(C = C_CONST)
            result = const_polynomial(X_ARRAY, constPoly_params)
            @test result == fill(C_CONST, length(X_ARRAY))

        end

    end

    @testset "Combined models" begin

        @testset "peak_model" begin

            gaussian_params = GaussianParams(A = A, mu = MU, sigma = SIGMA)
            compton_params = ComptonParams(h = H, mu = MU, sigma = SIGMA)
            peak_params = PeakParams(gaussian = gaussian_params, compton = compton_params)
            result = peak_model(X_ARRAY, peak_params)
            @test result == peak_model.(X_ARRAY, Ref(peak_params))

            expected =
                gaussian(X_ARRAY, gaussian_params) .+ compton(X_ARRAY, compton_params)
            @test result == expected

        end

        @testset "background_model" begin

            quadPoly_params = QuadPolyParams(C = C_QUAD, mu = MU)
            linPoly_params = LinPolyParams(C = C_LIN, mu = MU)
            constPoly_params = ConstPolyParams(C = C_CONST)
            background_params = BackgroundParams(
                quadPoly = quadPoly_params,
                linPoly = linPoly_params,
                constPoly = constPoly_params,
            )
            result = background_model(X_ARRAY, background_params)
            @test result == background_model.(X_ARRAY, Ref(background_params))

            expected =
                quad_polynomial(X_ARRAY, quadPoly_params) .+
                lin_polynomial(X_ARRAY, linPoly_params) .+
                const_polynomial(X_ARRAY, constPoly_params)
            @test result == expected

        end

        @testset "full_model" begin

            gaussian_params = GaussianParams(A = A, mu = MU, sigma = SIGMA)
            quadPoly_params = QuadPolyParams(C = C_QUAD, mu = MU)
            peak_params = PeakParams(gaussian = gaussian_params)
            background_params = BackgroundParams(quadPoly = quadPoly_params)
            model_params = ModelParams(peak = peak_params, background = background_params)
            result = full_model(X_ARRAY, model_params)
            @test result == full_model.(X_ARRAY, Ref(model_params))

            expected =
                peak_model(X_ARRAY, peak_params) .+
                background_model(X_ARRAY, background_params)
            @test result == expected

        end

    end

    @testset "Bool sentinel guards" begin

        @testset "peak_model ignores false" begin

            peak_params = PeakParams()
            @test peak_model(X_ARRAY, peak_params) == fill(0.0, length(X_ARRAY))

        end

        @testset "background_model ignores false" begin

            backround_params = BackgroundParams()
            @test background_model(X_ARRAY, backround_params) == fill(0.0, length(X_ARRAY))

        end

        @testset "full_model ignores nothing containers" begin

            model_params = ModelParams()
            @test full_model(X_ARRAY, model_params) == fill(0.0, length(X_ARRAY))

        end

    end
    
end
