using StatsBase: Histogram

@testset "models" begin

    MU = 2048.0
    SIGMA = 10.0
    C_CONST = 100.0
    BIN_SIZE = 0.5
    EDGES = (MU-3*SIGMA):BIN_SIZE:(MU+3*SIGMA)

    @testset "expected_counts" begin
        background = PolynomialShape(C_CONST)
        expected = fill(C_CONST * BIN_SIZE, length(EDGES) - 1)

        @test expected_counts(background, EDGES) ≈ expected
        @test expected_counts(background, Histogram(EDGES, zeros(Int, length(EDGES) - 1))) ≈
              expected
        # Non-uniform edges are allowed for the edges method
        @test expected_counts(background, [0.0, 1.0, 3.0]) ≈ C_CONST .* [1.0, 2.0]

        @test_throws ArgumentError expected_counts(background, [MU])  # one edge
        @test_throws ArgumentError expected_counts(background, [MU, MU - 1.0])  # decreasing
        @test_throws ArgumentError expected_counts(  # uneven, via _check_spectrum
            background,
            Histogram([0.0, 1.0, 3.0], [1, 2]),
        )
    end

end
