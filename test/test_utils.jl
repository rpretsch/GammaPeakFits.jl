using StatsBase: Histogram, mean
using GammaPeakFits: get_peak_features, mean_background, background_poly_bounds

@testset "utils" begin

    MU = 5.0
    BIN_SIZE = 1.0
    DATA = Histogram(0.5:BIN_SIZE:10.5, collect(1:10))  # bin centers 1.0:10.0
    COARSE_EDGES = 0.0:2.0:6.0  # bin centers 1.0, 3.0, 5.0, bin width 2.0

    @testset "cut_data" begin

        cut = cut_data(DATA; center = MU, width = 4.0)
        @test only(cut.edges) == [3.5, 4.5, 5.5, 6.5]
        @test cut.weights == [4, 5, 6]
        @test cut isa Histogram

        # Window outside the data contains no complete bin
        @test_throws ArgumentError cut_data(DATA; center = 50.0, width = 4.0)

    end

    @testset "mean_background" begin

        @test mean_background([2, 4], 2.0) == 1.5
        # Floored at one count per bin: one count / 0.5 keV bin = 2 counts/keV
        @test mean_background([0, 0], 0.5) == 2.0

    end

    @testset "background_poly_bounds" begin

        # h = 2 keV; b = 4 counts/keV
        lin_bound, quad_bound = background_poly_bounds(4.0, 2.0)
        @test lin_bound == 2 * 4.0 / 2
        @test quad_bound == 2 * 4.0 / 2^2

    end

    @testset "get_peak_features" begin

        @testset "Peak height and Peak area" begin
            height, area, background = get_peak_features(DATA; mu = 5.0, sigma = 1.0)
            @test height == 8.0 - mean([1.0, 9.0, 10.0])
            @test area == sqrt(2 * pi) * 1.0 * height / 1.0
            @test background == mean([1.0, 9.0, 10.0])
            @test height isa Float64
            @test area isa Float64
            @test background isa Float64
        end

        @testset "Converts counts/bin to counts/keV via bin_size" begin
            data = Histogram(COARSE_EDGES, [10, 100, 30])
            height, area, background = get_peak_features(data; mu = 3.0, sigma = 0.5)
            @test height == (100.0 - mean([10.0, 30.0])) / 2.0
            @test area == sqrt(2 * pi) * 0.5 * height
            @test background == mean([10.0, 30.0]) / 2.0
        end

        @testset "Sanity floors keep estimates positive" begin
            data = Histogram(COARSE_EDGES, [10, 20, 30])
            height, area, background = get_peak_features(data; mu = 3.0, sigma = 0.5)
            @test height == 1.0 / 2.0
            @test area == sqrt(2 * pi) * 0.5 * height
            @test background == mean([10.0, 30.0]) / 2.0
            @test height > 0
            @test area > eps()
            @test background > 0
        end

        @testset "Throws for non-positive sigma" begin
            @test_throws ArgumentError get_peak_features(DATA; mu = 5.0, sigma = 0.0)
            @test_throws ArgumentError get_peak_features(DATA; mu = 5.0, sigma = -1.0)
        end

        @testset "Throws when peak region not contained in data" begin
            data = Histogram(COARSE_EDGES, [10, 20, 30])
            @test_throws ArgumentError get_peak_features(data; mu = 2.0, sigma = 1.0)
        end

        @testset "Throws when peak region contains no bin centers" begin
            data = Histogram(COARSE_EDGES, [10, 20, 30])
            @test_throws ArgumentError get_peak_features(data; mu = 2.0, sigma = 0.1)
        end

        @testset "Throws when only peak region is contained in data" begin
            data = Histogram(COARSE_EDGES, [10, 20, 30])
            # `mu +- 3 * sigma` covers [1, 5] exactly, the data range
            @test_throws ArgumentError get_peak_features(data; mu = 3.0, sigma = 2 / 3)
        end

    end

end
