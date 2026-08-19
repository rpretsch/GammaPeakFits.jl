@testset "params" begin

    MU = 2048.0
    SIGMA = 10.0
    A = 1000.0
    H = 50.0
    TAU = 0.1
    IS_LOWENERGYTAIL = true
    C_QUAD = 1.0
    C_LIN = 10.0
    C_CONST = 100.0

    @testset "Component parameter structs" begin

        @testset "GaussianParams" begin
            gaussian_params = GaussianParams(A = A, mu = MU, sigma = SIGMA)
            @test gaussian_params.A == A
            @test gaussian_params.mu == MU
            @test gaussian_params.sigma == SIGMA
            @test gaussian_params isa GaussianParams
        end

        @testset "ComptonParams" begin
            compton_params = ComptonParams(h = H, mu = MU, sigma = SIGMA)
            @test compton_params.h == H
            @test compton_params.mu == MU
            @test compton_params.sigma == SIGMA
            @test compton_params isa ComptonParams
        end

        @testset "ExGaussianParams" begin
            exGaussian_params = ExGaussianParams(
                A = A,
                tau = TAU,
                is_lowEnergyTail = IS_LOWENERGYTAIL,
                mu = MU,
                sigma = SIGMA,
            )
            @test exGaussian_params.A == A
            @test exGaussian_params.tau == TAU
            @test exGaussian_params.is_lowEnergyTail == IS_LOWENERGYTAIL
            @test exGaussian_params.mu == MU
            @test exGaussian_params.sigma == SIGMA
            @test exGaussian_params isa ExGaussianParams
        end

        @testset "QuadPolyParams" begin
            quadPoly_params = QuadPolyParams(C = C_QUAD, mu = MU)
            @test quadPoly_params.C == C_QUAD
            @test quadPoly_params.mu == MU
            @test quadPoly_params isa QuadPolyParams
        end

        @testset "LinPolyParams" begin
            linPoly_params = LinPolyParams(C = C_LIN, mu = MU)
            @test linPoly_params.C == C_LIN
            @test linPoly_params.mu == MU
            @test linPoly_params isa LinPolyParams
        end

        @testset "ConstPolyParams" begin
            constPoly_params = ConstPolyParams(C = C_CONST)
            @test constPoly_params.C == C_CONST
            @test constPoly_params isa ConstPolyParams
        end

    end

    @testset "Container structs" begin

        @testset "PeakParams" begin
            peak_params = PeakParams()
            @test peak_params.gaussian isa Disabled
            @test peak_params.compton isa Disabled
            @test peak_params.lowEnergyTail isa Disabled
            @test peak_params.highEnergyTail isa Disabled

            peak_params = PeakParams(gaussian = Enabled())
            @test peak_params.gaussian isa Enabled
            @test peak_params.compton isa Disabled
            @test peak_params.lowEnergyTail isa Disabled
            @test peak_params.highEnergyTail isa Disabled

            peak_params = PeakParams(compton = ComptonParams(h = H, mu = MU, sigma = SIGMA))
            @test peak_params.gaussian isa Disabled
            @test peak_params.compton isa ComptonParams
            @test peak_params.lowEnergyTail isa Disabled
            @test peak_params.highEnergyTail isa Disabled

        end

        @testset "BackgroundParams" begin

            background_params = BackgroundParams()
            @test background_params.quadPoly isa Disabled
            @test background_params.linPoly isa Disabled
            @test background_params.constPoly isa Disabled

            background_params = BackgroundParams(quadPoly = Enabled())
            @test background_params.quadPoly isa Enabled
            @test background_params.linPoly isa Disabled
            @test background_params.constPoly isa Disabled

            background_params =
                BackgroundParams(linPoly = LinPolyParams(C = C_LIN, mu = MU))
            @test background_params.quadPoly isa Disabled
            @test background_params.linPoly isa LinPolyParams
            @test background_params.constPoly isa Disabled

        end

        @testset "ModelParams" begin

            model_params = ModelParams()
            @test model_params.peak isa Disabled
            @test model_params.background isa Disabled

            model_params = ModelParams(peak = PeakParams())
            @test model_params.peak isa PeakParams
            @test model_params.background isa Disabled

            @testset "NamedTuple constructor" begin

                model = ModelParams((gaussian_A = A, mu = MU, sigma = SIGMA))
                @test model.peak isa PeakParams
                @test model.peak.gaussian isa GaussianParams
                @test model.peak.gaussian.A == A
                @test model.peak.gaussian.mu == MU
                @test model.peak.gaussian.sigma == SIGMA
                @test model.peak.compton isa Disabled
                @test model.peak.lowEnergyTail isa Disabled
                @test model.peak.highEnergyTail isa Disabled
                @test model.background isa Disabled

                @testset "Throws on missing mu" begin
                    @test_throws ArgumentError ModelParams((gaussian_A = A, sigma = SIGMA))
                end

                @testset "Throws on missing sigma" begin
                    @test_throws ArgumentError ModelParams((gaussian_A = A, mu = MU))
                end

                @testset "Throws on partial low-energy tail pair" begin
                    @test_throws ArgumentError ModelParams((
                        lowEnergyTail_A = A,
                        mu = MU,
                        sigma = SIGMA,
                    ),)
                    @test_throws ArgumentError ModelParams((
                        lowEnergyTail_tau = TAU,
                        mu = MU,
                        sigma = SIGMA,
                    ),)
                end

                @testset "Throws on partial high-energy tail pair" begin
                    @test_throws ArgumentError ModelParams((
                        highEnergyTail_A = A,
                        mu = MU,
                        sigma = SIGMA,
                    ),)
                    @test_throws ArgumentError ModelParams((
                        highEnergyTail_tau = TAU,
                        mu = MU,
                        sigma = SIGMA,
                    ),)
                end

            end

        end

    end

end
