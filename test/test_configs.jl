@testset "configs" begin

    @testset "FitConfigs" begin

        configs = FitConfigs(mu = 2048.0, sigma = 5.0)
        @test configs.mu == 2048.0
        @test configs.sigma == 5.0
        @test configs.window_size == 50.0
        @test configs isa FitConfigs
        @test configs.integration_method isa Analytical

        custom_window = FitConfigs(mu = 2048.0, sigma = 5.0, window_size = 100.0)
        @test custom_window.window_size == 100.0

    end

end
