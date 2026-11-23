using StatsBase: Histogram

@testset "data" begin

    MU = 2048.0
    SIGMA = 10.0
    A = 1000.0
    BIN_SIZE = 0.5
    ENERGY_RANGE = (0.0, 4096.0)  # keV, outer bin edges
    NBINS = round(Int, (ENERGY_RANGE[2] - ENERGY_RANGE[1]) / BIN_SIZE)
    UNEVEN_EDGES = [0.0, 1.0, 3.0]
    UNEVEN_HIST = Histogram(UNEVEN_EDGES, [1, 2])

    @testset "_check_edges" begin
        check = GammaPeakFits._check_edges

        @testset "valid edges" begin
            @test check(0.0:BIN_SIZE:2.0) == BIN_SIZE
            @test check([0.0, 1.0]) == 1.0  # single bin
            @test check(collect(range(1000.0; step = 0.1, length = 101))) ≈ 0.1  # float-rounded
        end

        @testset "throws" begin
            @test_throws ArgumentError check([0.0])  # no bins
            @test_throws ArgumentError check(UNEVEN_EDGES)  # uneven
            @test_throws ArgumentError check([2.0, 1.0, 0.0])  # decreasing
            @test_throws ArgumentError check([0.0, 2.0, 1.0])  # up and down
        end
    end

    @testset "_check_spectrum" begin
        check = GammaPeakFits._check_spectrum
        edges = 0.0:1.0:2.0

        @testset "valid spectra" begin
            @test check(Histogram(0.0:BIN_SIZE:2.0, [1, 2, 3, 4])) == BIN_SIZE
            # Float weights that are whole numbers, e.g. from simulations
            @test check(Histogram(edges, [1.0, 2.0])) == step(edges)
        end

        @testset "throws" begin
            @test_throws ArgumentError check(Histogram(edges, [1.0, 2.0], :left, true))  # density
            @test_throws ArgumentError check(Histogram(edges, [1.5, 2.0]))  # not whole
            @test_throws ArgumentError check(Histogram(edges, [-1, 2]))  # negative
            @test_throws ArgumentError check(UNEVEN_HIST)  # edges, via _check_edges
        end
    end

    @testset "_uniform_edges" begin
        edges = GammaPeakFits._uniform_edges

        @testset "edges" begin
            @test edges(0.0, 2.0, BIN_SIZE) == [0.0, 0.5, 1.0, 1.5, 2.0]
        end

        @testset "float-rounded bin size" begin
            # No lost bin, last edge exact
            @test length(edges(1000.0, 1010.0, 0.1)) == 101
            @test last(edges(1000.0, 1010.0, 0.1)) == 1010.0
        end

        @testset "throws" begin
            @test_throws ArgumentError edges(0.0, 1.0, 0.0)  # bin size not positive
            @test_throws ArgumentError edges(0.0, 1.0, 0.3)  # no whole multiple
        end
    end

    @testset "sample_spectrum" begin

        @testset "keyword method" begin
            data = sample_spectrum(
                A * GaussPeak(MU, SIGMA);
                energy_range = ENERGY_RANGE,
                bin_size = BIN_SIZE,
            )
            @test length(only(data.edges)) == NBINS + 1
            @test data.weights isa Vector{Int64}
        end

        @testset "edges method" begin
            edges = range(ENERGY_RANGE...; length = NBINS + 1)
            data = sample_spectrum(A * GaussPeak(MU, SIGMA), edges)
            @test only(data.edges) == edges
        end

        @testset "throws" begin
            # Negative expected counts
            @test_throws ArgumentError sample_spectrum(
                PolynomialShape(-1.0);
                bin_size = BIN_SIZE,
                energy_range = ENERGY_RANGE,
            )
            # Non-finite expected counts
            @test_throws ArgumentError sample_spectrum(
                Inf * GaussPeak(MU, SIGMA);
                bin_size = BIN_SIZE,
                energy_range = ENERGY_RANGE,
            )
            # Uneven edges, rejected by _check_edges in expected_counts
            @test_throws ArgumentError sample_spectrum(PolynomialShape(1.0), UNEVEN_EDGES)
            # Energy range is no whole multiple of bin_size, rejected by _uniform_edges
            @test_throws ArgumentError sample_spectrum(
                PolynomialShape(1.0);
                bin_size = 0.3,
                energy_range = (0.0, 1.0),
            )
        end

    end

    @testset "bin_energies" begin
        energies = [0.2, 0.7, 1.1, 1.9, 2.5]

        @testset "derived energy range" begin
            data = bin_energies(energies, 1.0)
            @test only(data.edges) == [0.0, 1.0, 2.0, 3.0]
            @test data.weights == [2, 2, 1]
        end

        @testset "explicit energy range drops energies outside" begin
            data = bin_energies(energies, 0.5; energy_range = (0.5, 2.0))
            @test only(data.edges) == [0.5, 1.0, 1.5, 2.0]
            @test data.weights == [1, 1, 1]
        end

        @testset "throws" begin
            @test_throws ArgumentError bin_energies(energies, 0.0)  # bin size not positive
            # Energy range is no whole multiple of bin_size, rejected by _uniform_edges
            @test_throws ArgumentError bin_energies(
                energies,
                0.3;
                energy_range = (0.0, 1.0),
            )
            # No energies to derive the range from
            @test_throws ArgumentError bin_energies(Float64[], 1.0)
        end
    end

    @testset "rebin" begin
        data = Histogram(0.0:1.0:5.0, [1, 2, 3, 4, 5])

        @testset "merges neighbouring bins, drops the incomplete rest" begin
            merged = rebin(data, 2.0)
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
            # uneven bins, via _check_spectrum
            @test_throws ArgumentError rebin(UNEVEN_HIST, 2.0)
        end
    end

end
