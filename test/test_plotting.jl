using CairoMakie: Makie

@testset "plotting" begin

    DATA = SpectrumData(collect(1.0:10.0), collect(1:10))

    @testset "plot_data" begin

        @testset "Returns a (fig, ax) tuple" begin
            result = plot_data(DATA)
            @test result isa Tuple
            @test length(result) == 2
            fig, ax = result
            @test fig isa Makie.Figure
            @test ax isa Makie.Axis
        end

        @testset "Accepts optional mu marker" begin
            result = plot_data(DATA; mu = 5.0)
            @test result isa Tuple
            @test length(result) == 2
            fig, ax = result
            @test fig isa Makie.Figure
            @test ax isa Makie.Axis
        end

    end

end
