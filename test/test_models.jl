using StatsBase: Histogram

@testset "models" begin

    MU = 2048.0
    SIGMA = 10.0
    C_CONST = 100.0
    BIN_SIZE = 0.5
    EDGES = (MU-3*SIGMA):BIN_SIZE:(MU+3*SIGMA)
    UNEVEN_EDGES = [0.0, 1.0, 3.0]

    @testset "expected_counts" begin
        background = PolynomialShape(C_CONST)
        expected = fill(C_CONST * BIN_SIZE, length(EDGES) - 1)

        @test expected_counts(background, EDGES) ≈ expected
        @test expected_counts(background, Histogram(EDGES, zeros(Int, length(EDGES) - 1))) ≈
              expected
        # Uneven bins, rejected by _check_edges
        @test_throws ArgumentError expected_counts(background, UNEVEN_EDGES)
    end

    @testset "densityof" begin
        # Re-exported; the density of a composite model is in counts/keV
        peak_counts = 1000.0
        model = peak_counts * GaussPeak(MU, SIGMA) + PolynomialShape(C_CONST)
        @test densityof(model, MU) ≈ peak_counts / (sqrt(2 * pi) * SIGMA) + C_CONST
    end

end
