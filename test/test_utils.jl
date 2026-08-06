@testset "utils" begin

    DATA = SpectrumData(
        bin_centers = collect(1.0:10.0),
        weights = collect(1:10),
        bin_size = 1.0,
    )

    @testset "cut_data" begin

        cut = cut_data(DATA, 5.0, 4.0)
        @test cut.bin_centers == [3.0, 4.0, 5.0, 6.0, 7.0]
        @test cut.weights == [3, 4, 5, 6, 7]
        @test cut.bin_size == 1.0
        @test cut isa SpectrumData{Float64,Int}

    end

    @testset "get_peak_features" begin

        @testset "Peak height and Peak area" begin
            height, area = get_peak_features(DATA, 5.0, 1.0)
            @test height == 8.0   # max weight in [2, 8] = 8, divided by bin_size 1.0
            @test area == 48.0    # 6 * sigma * max = 6 * 1.0 * 8 / 1.0
            @test height isa Float64
            @test area isa Float64
        end

        @testset "Converts counts/bin to counts/keV via bin_size" begin
            data = SpectrumData(
                bin_centers = [1.0, 3.0, 5.0],
                weights = [10, 20, 30],
                bin_size = 2.0,
            )
            height, area = get_peak_features(data, 3.0, 1.0)
            @test height == 15.0   # 30 counts/bin / 2.0 keV/bin
            @test area == 90.0     # 6 * 1.0 * 30 counts/bin / 2.0 keV/bin
        end

    end

end
