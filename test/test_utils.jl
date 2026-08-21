using StatsBase: mean
using GammaPeakFits: is_present

@testset "utils" begin

    DATA = SpectrumData(collect(1.0:10.0), collect(1:10))

    @testset "cut_data" begin

        cut = cut_data(DATA, 5.0, 4.0)
        @test cut.bin_centers == [4.0, 5.0, 6.0]
        @test cut.weights == [4, 5, 6]
        @test cut.bin_size == 1.0
        @test cut isa SpectrumData

    end

    @testset "is_present" begin

        @test !is_present(Disabled())
        @test is_present(Enabled())

    end

    @testset "get_peak_features" begin

        @testset "Peak height and Peak area" begin
            height, area, background = get_peak_features(DATA, 5.0, 1.0)
            @test height == 8.0 - mean([1.0, 9.0, 10.0])
            @test area == sqrt(2 * pi) * 1.0 * height / 1.0
            @test background == mean([1.0, 9.0, 10.0])
            @test height isa Float64
            @test area isa Float64
            @test background isa Float64
        end

        @testset "Converts counts/bin to counts/keV via bin_size" begin
            data = SpectrumData([1.0, 3.0, 5.0], [10, 20, 30])
            height, area, background = get_peak_features(data, 3.0, 0.5)
            @test height == (20.0 - mean([10.0, 30.0])) / 2.0
            @test area == sqrt(2 * pi) * 0.5 * height / 2.0
            @test background == mean([10.0, 30.0]) / 2.0
        end

        @testset "Throws when peak region not contained in data" begin
            data = SpectrumData([1.0, 3.0, 5.0], [10, 20, 30])
            @test_throws ArgumentError get_peak_features(data, 2.0, 1.0)
        end

        @testset "Throws when only peak region is contained in data" begin
            data = SpectrumData([1.0, 3.0, 5.0], [10, 20, 30])
            @test_throws ArgumentError get_peak_features(data, 2.0, 1.0)
        end

    end

end
