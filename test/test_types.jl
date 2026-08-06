@testset "types" begin

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
            @test gaussian_params isa GaussianParams{Float64}
        end

        @testset "ComptonParams" begin
            compton_params = ComptonParams(h = H, mu = MU, sigma = SIGMA)
            @test compton_params.h == H
            @test compton_params.mu == MU
            @test compton_params.sigma == SIGMA
            @test compton_params isa ComptonParams{Float64}
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
            @test exGaussian_params isa ExGaussianParams{Float64}
        end

        @testset "QuadPolyParams" begin
            quadPoly_params = QuadPolyParams(C = C_QUAD, mu = MU)
            @test quadPoly_params.C == C_QUAD
            @test quadPoly_params.mu == MU
            @test quadPoly_params isa QuadPolyParams{Float64}
        end

        @testset "LinPolyParams" begin
            linPoly_params = LinPolyParams(C = C_LIN, mu = MU)
            @test linPoly_params.C == C_LIN
            @test linPoly_params.mu == MU
            @test linPoly_params isa LinPolyParams{Float64}
        end

        @testset "ConstPolyParams" begin
            constPoly_params = ConstPolyParams(C = C_CONST)
            @test constPoly_params.C == C_CONST
            @test constPoly_params isa ConstPolyParams{Float64}
        end

    end

    @testset "Container structs" begin

        @testset "PeakParams" begin
            peak_params = PeakParams()
            @test peak_params.gaussian === false
            @test peak_params.compton === false
            @test peak_params.lowEnergyTail === false
            @test peak_params.highEnergyTail === false

            peak_params = PeakParams(gaussian = true)
            @test peak_params.gaussian === true
            @test peak_params.compton === false
            @test peak_params.lowEnergyTail === false
            @test peak_params.highEnergyTail === false

            peak_params = PeakParams(compton = ComptonParams(h = H, mu = MU, sigma = SIGMA))
            @test peak_params.gaussian === false
            @test peak_params.compton isa ComptonParams
            @test peak_params.lowEnergyTail === false
            @test peak_params.highEnergyTail === false

        end

        @testset "BackgroundParams" begin

            background_params = BackgroundParams()
            @test background_params.quadPoly === false
            @test background_params.linPoly === false
            @test background_params.constPoly === false

            background_params = BackgroundParams(quadPoly = true)
            @test background_params.quadPoly === true
            @test background_params.linPoly === false
            @test background_params.constPoly === false

            background_params =
                BackgroundParams(linPoly = LinPolyParams(C = C_LIN, mu = MU))
            @test background_params.quadPoly === false
            @test background_params.linPoly isa LinPolyParams
            @test background_params.constPoly === false

        end

        @testset "ModelParams" begin

            model_params = ModelParams()
            @test isnothing(model_params.peak)
            @test isnothing(model_params.background)

            model_params = ModelParams(peak = PeakParams())
            @test model_params.peak isa PeakParams
            @test isnothing(model_params.background)

        end

        @testset "SpectrumData" begin

            @testset "inner constructor" begin

                data = SpectrumData(
                    bin_centers = collect(1.0:4096.0),
                    weights = zeros(Int64, 4096),
                    bin_size = 1.0,
                )
                @test data.bin_centers isa AbstractVector{Float64}
                @test data.weights isa AbstractVector{Int64}
                @test data.bin_size == 1.0
                @test data isa SpectrumData{Float64,Int64}

            end

            @testset "outer constructor" begin

                gaussian_params = GaussianParams(A = A, mu = MU, sigma = SIGMA)
                model_params = ModelParams(peak = PeakParams(gaussian = gaussian_params))

                data = SpectrumData(1.0, 4096.0, 0.5, model_params)
                @test data.bin_centers isa AbstractVector{Float64}
                @test length(data.bin_centers) == 8191
                @test length(data.weights) == 8191
                @test data.weights isa AbstractVector{Int64}
                @test data.bin_size == 0.5
                @test data isa SpectrumData{Float64,Int64}

            end

            @testset "outer constructor throws on negative expected counts" begin

                model_params = ModelParams(
                    background = BackgroundParams(
                        linPoly = LinPolyParams(C = -1.0, mu = 0.0),
                    ),
                )
                @test_throws ArgumentError SpectrumData(1.0, 10.0, 1.0, model_params)

            end

        end

    end

end
