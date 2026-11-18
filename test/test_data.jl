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

        @testset "outer constructor throws on non-finite expected counts" begin

            model_params = ModelParams(
                peak = PeakParams(
                    gaussian = GaussianParams(A = 1.0e308, mu = MU, sigma = 0.1),
                ),
            )
            @test_throws ArgumentError SpectrumData(2040.0, 2050.0, 1.0, model_params)

        end

        @testset "markers constructor" begin

            @testset "bin centers markers" begin
                data = SpectrumData(collect(1.0:4.0), [10, 20, 30, 40])
                @test data.bin_centers == collect(1.0:4.0)
                @test data.bin_edges == collect(0.5:1.0:4.5)
                @test data.weights == [10, 20, 30, 40]
                @test data.bin_size == 1.0
            end

            @testset "bin edges markers" begin
                edges = collect(0.5:1.0:4.5)
                data = SpectrumData(edges, [10, 20, 30, 40])
                @test data.bin_edges == edges
                @test data.bin_centers == collect(1.0:1.0:4.0)
                @test data.weights == [10, 20, 30, 40]
                @test data.bin_size == 1.0
            end

            @testset "explicit bin_size with single marker" begin
                data = SpectrumData([5.0], [1]; bin_size = 1.0)
                @test data.bin_centers == [5.0]
                @test data.bin_edges == collect(4.5:1.0:5.5)
                @test data.bin_size == 1.0
            end

            @testset "float-rounded bin markers" begin
                markers = collect(range(1000.0; step = 0.1, length = 101))

                data = SpectrumData(markers, collect(1:100))
                @test length(data.bin_centers) == 100
                @test data.bin_size ≈ 0.1
                @test data.bin_centers ≈ markers[1:(end-1)] .+ 0.05

                data = SpectrumData(markers, collect(1:101))
                @test length(data.bin_edges) == 102
                @test data.bin_size ≈ 0.1
                @test data.bin_edges ≈ [markers .- 0.05; markers[end] + 0.05]
            end

            @testset "throws on mismatched marker/weight sizes" begin
                @test_throws ArgumentError SpectrumData(collect(1.0:4.0), [1, 2])
            end

            @testset "throws on single marker without bin_size" begin
                @test_throws ArgumentError SpectrumData([1.0], [1])
            end

            @testset "throws on non-uniformly spaced bins" begin
                @test_throws ArgumentError SpectrumData([1.0, 2.0, 4.0], [1, 2, 3])
            end

            @testset "throws on bin_size not matching the markers" begin
                @test_throws ArgumentError SpectrumData(
                    collect(1.0:4.0),
                    [1, 2, 3, 4];
                    bin_size = 2.0,
                )
            end

        end

        @testset "bin_energies" begin
            energies = [0.2, 0.7, 1.1, 1.9, 2.5]

            @testset "derived energy range" begin
                data = bin_energies(energies, 1.0)
                @test data.bin_edges == [0.0, 1.0, 2.0, 3.0]
                @test data.weights == [2, 2, 1]
                @test data.bin_size == 1.0
            end

            @testset "explicit energy range drops energies outside" begin
                data = bin_energies(energies, 0.5; energy_range = (0.5, 2.0))
                @test data.bin_edges == [0.5, 1.0, 1.5, 2.0]
                @test data.weights == [1, 1, 1]
            end

            @testset "float-rounded bin size" begin
                data = bin_energies(fill(1000.05, 3), 0.1; energy_range = (1000.0, 1010.0))
                @test length(data.weights) == 100
                @test data.weights[1] == 3
            end

            @testset "throws" begin
                @test_throws ArgumentError bin_energies(energies, 0.0)
                @test_throws ArgumentError bin_energies(
                    energies,
                    0.3;
                    energy_range = (0.0, 1.0),
                )
                @test_throws ArgumentError bin_energies(Float64[], 1.0)
            end
        end

        @testset "rebin" begin
            data = SpectrumData(collect(0.0:1.0:5.0), [1, 2, 3, 4, 5])

            @testset "merges neighbouring bins, drops the incomplete rest" begin
                merged = rebin(data, 2.0)
                @test merged.bin_edges == [0.0, 2.0, 4.0]
                @test merged.weights == [3, 7]
                @test merged.bin_size == 2.0
            end

            @testset "same bin size keeps the spectrum" begin
                @test rebin(data, 1.0).weights == data.weights
            end

            @testset "throws" begin
                @test_throws ArgumentError rebin(data, 1.5)  # no whole multiple
                @test_throws ArgumentError rebin(data, 0.5)  # finer than the data
                @test_throws ArgumentError rebin(data, 6.0)  # wider than the spectrum
            end
        end

    end

end
