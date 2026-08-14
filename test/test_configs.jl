@testset "configs" begin

    @testset "PriorConfigs" begin

        configs = PriorConfigs()
        @test configs isa PriorConfigs{Float64}

    end

    @testset "FitConfigs" begin

        configs = FitConfigs(mu = 2048.0, sigma = 5.0)
        @test configs.mu == 2048.0
        @test configs.sigma == 5.0
        @test configs.prior == PriorConfigs()
        @test configs isa FitConfigs{Float64}

    end

end
