using StatsBase: Histogram

@testset "data" begin

    MU = 2048.0
    SIGMA = 10.0
    A = 1000.0
    BIN_SIZE = 0.5
    ENERGY_RANGE = (0.0, 4096.0)  # keV, outer bin edges
    NBINS = round(Int, (ENERGY_RANGE[2] - ENERGY_RANGE[1]) / BIN_SIZE)

    @testset "_check_spectrum" begin
        check = GammaPeakFits._check_spectrum

        @test check(Histogram(0.0:0.5:2.0, [1, 2, 3, 4])) == 0.5
        @test check(Histogram([0.0, 1.0], [0])) == 1.0  # single bin
        # Float weights that are whole numbers, e.g. from simulations
        @test check(Histogram(0.0:1.0:2.0, [1.0, 2.0])) == 1.0
        # Float-rounded edges
        @test check(
            Histogram(collect(range(1000.0; step = 0.1, length = 101)), ones(Int, 100)),
        ) ≈ 0.1

        @test_throws ArgumentError check(Histogram([0.0], Int[]))  # no bins
        @test_throws ArgumentError check(Histogram([0.0, 1.0, 3.0], [1, 2]))  # uneven
        @test_throws ArgumentError check(Histogram([2.0, 1.0, 0.0], [1, 2]))  # decreasing
        @test_throws ArgumentError check(Histogram(0.0:1.0:2.0, [1.0, 2.0], :left, true))
        # density
        @test_throws ArgumentError check(Histogram(0.0:1.0:2.0, [1.5, 2.0]))  # not whole
        @test_throws ArgumentError check(Histogram(0.0:1.0:2.0, [-1, 2]))  # negative
    end

    @testset "sample_spectrum" begin

        @testset "measure model returns a histogram of counts" begin
            data = sample_spectrum(
                A * GaussPeak(MU, SIGMA);
                energy_range = ENERGY_RANGE,
                bin_size = BIN_SIZE,
            )
            @test data isa Histogram
            @test length(only(data.edges)) == NBINS + 1
            @test length(data.weights) == NBINS
            @test data.weights isa Vector{Int64}
            @test GammaPeakFits._check_spectrum(data) ≈ BIN_SIZE
        end

        @testset "measure model throws on negative expected counts" begin
            @test_throws ArgumentError sample_spectrum(
                PolynomialShape(-1.0);
                bin_size = 1.0,
                energy_range = (0.5, 10.5),
            )
        end

        @testset "measure model on given edges" begin
            edges = range(ENERGY_RANGE...; length = NBINS + 1)
            data = sample_spectrum(A * GaussPeak(MU, SIGMA), edges)
            @test data isa Histogram
            @test only(data.edges) == edges
            @test length(data.weights) == NBINS

            # Non-uniform edges give no valid spectrum
            @test_throws ArgumentError sample_spectrum(
                PolynomialShape(1.0),
                [0.0, 1.0, 3.0],
            )
        end

    end

    @testset "bin_energies" begin
        energies = [0.2, 0.7, 1.1, 1.9, 2.5]

        @testset "derived energy range" begin
            data = bin_energies(energies, 1.0)
            @test data isa Histogram
            @test only(data.edges) == [0.0, 1.0, 2.0, 3.0]
            @test data.weights == [2, 2, 1]
        end

        @testset "explicit energy range drops energies outside" begin
            data = bin_energies(energies, 0.5; energy_range = (0.5, 2.0))
            @test only(data.edges) == [0.5, 1.0, 1.5, 2.0]
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
        data = Histogram(0.0:1.0:5.0, [1, 2, 3, 4, 5])

        @testset "merges neighbouring bins, drops the incomplete rest" begin
            merged = rebin(data, 2.0)
            @test merged isa Histogram
            @test only(merged.edges) == [0.0, 2.0, 4.0]
            @test merged.weights == [3, 7]
        end

        @testset "same bin size keeps the spectrum" begin
            @test rebin(data, 1.0).weights == data.weights
        end

        @testset "throws" begin
            @test_throws ArgumentError rebin(data, 1.5)  # no whole multiple
            @test_throws ArgumentError rebin(data, 0.5)  # finer than the data
            @test_throws ArgumentError rebin(data, 6.0)  # wider than the spectrum
            @test_throws ArgumentError rebin(Histogram([0.0, 1.0, 3.0], [1, 2]), 2.0)  # uneven
        end
    end

end
