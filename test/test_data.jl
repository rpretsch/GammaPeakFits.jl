@testset "data" begin

    MU = 2048.0
    SIGMA = 10.0
    A = 1000.0

    @testset "SpectrumData" begin

        @testset "inner constructor" begin

            data = SpectrumData(
                bin_centers = collect(1.0:4096.0),
                bin_edges = collect(0.5:4096.5),
                weights = zeros(Int64, 4096),
                bin_size = 1.0,
            )
            @test data.bin_centers isa Vector{Float64}
            @test data.bin_centers == collect(1.0:4096.0)
            @test data.bin_edges isa Vector{Float64}
            @test data.bin_edges == collect(0.5:4096.5)
            @test data.weights isa Vector{Int64}
            @test data.weights == zeros(Int64, 4096)
            @test data.bin_size == 1.0
            @test length(data.bin_edges) == (length(data.bin_centers) + 1)
            @test data isa SpectrumData

        end

        @testset "outer constructor" begin

            gaussian_params = GaussianParams(A = A, mu = MU, sigma = SIGMA)
            model_params = ModelParams(peak = PeakParams(gaussian = gaussian_params))

            data = SpectrumData(1.0, 4096.0, 0.5, model_params)
            @test data.bin_centers isa Vector{Float64}
            @test length(data.bin_centers) == 8191
            @test length(data.bin_edges) == 8192
            @test length(data.weights) == 8191
            @test data.weights isa Vector{Int64}
            @test data.bin_size == 0.5
            @test data isa SpectrumData

        end

        @testset "outer constructor throws on negative expected counts" begin

            model_params = ModelParams(
                background = BackgroundParams(linPoly = LinPolyParams(C = -1.0, mu = 0.0)),
            )
            @test_throws ArgumentError SpectrumData(1.0, 10.0, 1.0, model_params)

        end

    end

end
