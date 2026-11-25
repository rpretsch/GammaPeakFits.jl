using StatsBase: Histogram, mean, midpoints
using GammaPeakFits:
    _get_peak_features,
    _mean_background,
    _background_poly_bounds,
    _peak_params,
    _peak_mask,
    _background_mask

@testset "utils" begin

    MU = 5.0
    SIGMA = 0.6
    BIN_SIZE = 1.0
    # Bin centers 1.0:10.0: peak at MU, 2 counts/bin outside MU ± 5 SIGMA (bins 1, 9, 10)
    DATA = Histogram(0.5:BIN_SIZE:10.5, [2, 3, 3, 5, 12, 5, 3, 3, 2, 2])
    MUS = [3.0, 8.0]  # two peak centroids within DATA
    UNEVEN_HIST = Histogram([0.0, 1.0, 3.0], [1, 2])

    @testset "cut_data" begin

        @testset "keeps the bins inside the window" begin
            cut = cut_data(DATA; center = MU, width = 4.0)
            @test only(cut.edges) == [3.5, 4.5, 5.5, 6.5]
            @test cut.weights == [5, 12, 5]
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

    @testset "_peak_params" begin

        @testset "one peak" begin
            @test _peak_params(MU, SIGMA) == ([MU], [SIGMA])
        end

        @testset "several peaks with a shared or an own sigma" begin
            @test _peak_params(MUS, SIGMA) == (MUS, [SIGMA, SIGMA])
            @test _peak_params(MUS, [0.5, 0.7]) == (MUS, [0.5, 0.7])
        end

        @testset "throws" begin
            # No peak
            @test_throws ArgumentError _peak_params(Float64[], SIGMA)
            # Two widths for three peaks
            @test_throws ArgumentError _peak_params([1.0, 2.0, 3.0], [0.5, 0.5])
            # sigma not positive
            @test_throws ArgumentError _peak_params(MU, 0.0)
            @test_throws ArgumentError _peak_params(MUS, [0.5, -1.0])
        end

    end

    @testset "_peak_mask" begin
        centers = midpoints(only(DATA.edges))
        @test findall(_peak_mask(centers; mu = MU, sigma = SIGMA)) == 4:6
        # Peak region [4.2, 4.8] lies between two bin centers
        @test_throws ArgumentError _peak_mask(centers; mu = 4.5, sigma = 0.1)
    end

    @testset "_background_mask" begin
        centers = midpoints(only(DATA.edges))

        @testset "selects the bins outside mu ± 5 sigma" begin
            mask = _background_mask(centers; mu = [MU], sigma = [SIGMA])
            @test findall(mask) == [1, 9, 10]
        end

        @testset "excludes every peak" begin
            # mu ± 5 sigma: [2, 4] and [7, 9]
            mask = _background_mask(centers; mu = MUS, sigma = [0.2, 0.2])
            @test findall(mask) == [1, 5, 6, 10]
        end

        @testset "throws" begin
            # mu ± 5 sigma = [0, 10] covers all bin centers
            @test_throws ArgumentError _background_mask(centers; mu = [MU], sigma = [1.0])
        end
    end

    @testset "_get_peak_features" begin

        @testset "peak height and area" begin
            height, area, background = _get_peak_features(DATA; mu = MU, sigma = SIGMA)
            # Bins 2, 3, 7, 8 lie between 3 and 5 sigma and are not part of the background
            @test background == mean(DATA.weights[[1, 9, 10]]) / BIN_SIZE
            @test height == maximum(DATA.weights) / BIN_SIZE - background
            @test area == sqrt(2 * pi) * SIGMA * height
        end

        @testset "background on one side is enough" begin
            # mu ± 5 sigma = [4.5, 10.5]: background bins 1 to 4, all below the peak
            _, _, background = _get_peak_features(DATA; mu = 7.5, sigma = SIGMA)
            @test background == mean(DATA.weights[1:4]) / BIN_SIZE
        end

        coarse_edges = 0.0:2.0:14.0  # bin centers 1.0:2.0:13.0
        coarse_mu = 7.0
        coarse_sigma = 0.5
        coarse_bin_size = step(coarse_edges)
        coarse_background_bins = [1, 2, 6, 7]  # outside coarse_mu ± 5 coarse_sigma

        @testset "converts counts/bin to counts/keV via the bin size" begin
            data = Histogram(coarse_edges, [10, 10, 20, 100, 20, 30, 30])
            height, _, background =
                _get_peak_features(data; mu = coarse_mu, sigma = coarse_sigma)
            @test background == mean(data.weights[coarse_background_bins]) / coarse_bin_size
            @test height == maximum(data.weights) / coarse_bin_size - background
        end

        @testset "floors keep estimates positive" begin
            # The peak bin (15) lies below the background (20)
            data = Histogram(coarse_edges, [10, 10, 20, 15, 20, 30, 30])
            height, _, background =
                _get_peak_features(data; mu = coarse_mu, sigma = coarse_sigma)
            @test height == 1.0 / coarse_bin_size
            @test background == mean(data.weights[coarse_background_bins]) / coarse_bin_size
        end

        @testset "several peaks" begin
            # Bin centers 1.0:20.0, 2 counts/bin background, peaks at 6 and 14
            data = Histogram(
                0.5:BIN_SIZE:20.5,
                [2, 2, 2, 3, 5, 12, 5, 3, 2, 2, 2, 3, 5, 22, 5, 3, 2, 2, 2, 2],
            )

            @testset "background outside every peak" begin
                height, _, background = _get_peak_features(
                    data;
                    mu = [6.0, 14.0],
                    sigma = SIGMA,
                    peak_index = 2,
                )
                @test background == mean(data.weights[[1, 2, 10, 18, 19, 20]]) / BIN_SIZE
                @test height == maximum(data.weights) / BIN_SIZE - background
            end

            @testset "a peak at the window edge needs no background beyond it" begin
                _, _, background = _get_peak_features(data; mu = [6.0, 20.0], sigma = SIGMA)
                @test background == mean(data.weights[[1, 2, 10, 11, 12, 13, 14, 15, 16]])
            end
        end

        @testset "throws" begin
            # Uneven bins, rejected by _check_spectrum
            @test_throws ArgumentError _get_peak_features(
                UNEVEN_HIST;
                mu = 1.5,
                sigma = 0.1,
            )
            # sigma not positive, rejected by _peak_params
            @test_throws ArgumentError _get_peak_features(DATA; mu = MU, sigma = 0.0)
            # Second peak requested, but only one given
            @test_throws ArgumentError _get_peak_features(
                DATA;
                mu = MU,
                sigma = SIGMA,
                peak_index = 2,
            )
            # Peak region between two bin centers, rejected by _peak_mask
            @test_throws ArgumentError _get_peak_features(DATA; mu = 4.5, sigma = 0.1)
            # mu ± 5 sigma covers all bins, rejected by _background_mask
            @test_throws ArgumentError _get_peak_features(DATA; mu = MU, sigma = 1.0)
        end

    end

end
