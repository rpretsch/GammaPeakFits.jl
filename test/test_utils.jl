using StatsBase: Histogram, mean, midpoints
using GammaPeakFits:
    _get_peak_features, _mean_background, _background_poly_bounds, _peak_mask

@testset "utils" begin

    MU = 5.0
    SIGMA = 1.0
    BIN_SIZE = 1.0
    DATA = Histogram(0.5:BIN_SIZE:10.5, collect(1:10))  # bin centers 1.0:10.0
    COARSE_EDGES = 0.0:2.0:6.0
    UNEVEN_HIST = Histogram([0.0, 1.0, 3.0], [1, 2])

    @testset "cut_data" begin

        @testset "keeps the bins inside the window" begin
            cut = cut_data(DATA; center = MU, width = 4.0)
            @test only(cut.edges) == [3.5, 4.5, 5.5, 6.5]
            @test cut.weights == [4, 5, 6]
        end

        @testset "throws" begin
            # Window outside the data contains no complete bin
            @test_throws ArgumentError cut_data(DATA; center = 50.0, width = 4.0)
            # Uneven bins, rejected by _check_spectrum
            @test_throws ArgumentError cut_data(UNEVEN_HIST; center = 1.5, width = 3.0)
        end

    end

    @testset "_mean_background" begin

        weights = [2, 4]  # counts per bin
        bin_size = 2.0    # keV
        @test _mean_background(weights, bin_size) == mean(weights) / bin_size

        # Floored at one count per bin
        @test _mean_background(zeros(Int, 2), bin_size) == 1 / bin_size

    end

    @testset "_background_poly_bounds" begin

        level = 4.0        # mean background in counts/keV
        half_window = 2.0  # half-width of the fit window in keV
        lin_bound, quad_bound = _background_poly_bounds(level, half_window)
        @test lin_bound == 2 * level / half_window
        @test quad_bound == 2 * level / half_window^2

    end

    @testset "_peak_mask" begin
        centers = midpoints(only(DATA.edges))
        coarse_centers = midpoints(COARSE_EDGES)

        @testset "selects the bins within mu ± 3 sigma" begin
            @test findall(_peak_mask(centers; mu = MU, sigma = SIGMA)) == 2:8
        end

        @testset "throws" begin
            # sigma not positive
            @test_throws ArgumentError _peak_mask(centers; mu = MU, sigma = 0.0)
            @test_throws ArgumentError _peak_mask(centers; mu = MU, sigma = -1.0)
            # Peak region not contained in the data
            @test_throws ArgumentError _peak_mask(coarse_centers; mu = 2.0, sigma = 1.0)
            # Peak region contains no bin centers
            @test_throws ArgumentError _peak_mask(coarse_centers; mu = 2.0, sigma = 0.1)
            # `mu +- 3 * sigma` covers [1, 5] exactly, all bins are in the peak region
            @test_throws ArgumentError _peak_mask(coarse_centers; mu = 3.0, sigma = 2 / 3)
        end
    end

    @testset "_get_peak_features" begin

        coarse_mu = 3.0
        coarse_sigma = 0.5
        coarse_bin_size = step(COARSE_EDGES)

        @testset "peak height and area" begin
            height, area, background = _get_peak_features(DATA; mu = MU, sigma = SIGMA)
            @test height == 8.0 - mean([1.0, 9.0, 10.0])
            @test area == sqrt(2 * pi) * SIGMA * height
            @test background == mean([1.0, 9.0, 10.0])
        end

        @testset "converts counts/bin to counts/keV via the bin size" begin
            data = Histogram(COARSE_EDGES, [10, 100, 30])
            height, _, background =
                _get_peak_features(data; mu = coarse_mu, sigma = coarse_sigma)
            @test height == (100.0 - mean([10.0, 30.0])) / coarse_bin_size
            @test background == mean([10.0, 30.0]) / coarse_bin_size
        end

        @testset "floors keep estimates positive" begin
            data = Histogram(COARSE_EDGES, [10, 20, 30])
            height, _, background =
                _get_peak_features(data; mu = coarse_mu, sigma = coarse_sigma)
            @test height == 1.0 / coarse_bin_size
            @test background == mean([10.0, 30.0]) / coarse_bin_size
        end

        @testset "throws" begin
            # Uneven bins, rejected by _check_spectrum
            @test_throws ArgumentError _get_peak_features(
                UNEVEN_HIST;
                mu = 1.5,
                sigma = 0.1,
            )
            # sigma not positive, rejected by _peak_mask
            @test_throws ArgumentError _get_peak_features(DATA; mu = MU, sigma = 0.0)
        end

    end

end
