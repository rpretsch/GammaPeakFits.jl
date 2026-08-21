@testset "Quick start example (README / module docstring)" begin

    seed!(42)

    # Configurations
    configs = FitConfigs(
        mu = 2048.0,   # keV
        sigma = 5.0,   # keV
    )

    # Generate data
    A = 1000.0              # counts
    C_const = 100.0         # counts/keV
    lower_limit = 1.0       # keV
    upper_limit = 4096.0    # keV
    bin_size = 0.5          # keV

    gaussian_params = GaussianParams(A = A, mu = configs.mu, sigma = configs.sigma)
    peak_params = PeakParams(gaussian = gaussian_params)
    constPoly_params = ConstPolyParams(C = C_const)
    background_params = BackgroundParams(constPoly = constPoly_params)
    generation_modelParams = ModelParams(peak = peak_params, background = background_params)
    data = SpectrumData(lower_limit, upper_limit, bin_size, generation_modelParams)

    # cut appropriate fit window
    window_size = 100.0 # keV
    fit_data = cut_data(data, configs.mu, window_size)

    # Specify which components to include for fitting
    peak_params = PeakParams(gaussian = Enabled())
    background_params = BackgroundParams(constPoly = Enabled())
    fit_modelParams = ModelParams(peak = peak_params, background = background_params)

    # Build the prior / posterior
    prior = build_prior(fit_data, fit_modelParams, configs)
    posterior = build_posterior(fit_data, prior, configs)

    # Sample with BAT.jl
    result = bat_sample(
        posterior,
        TransformedMCMC(proposal = RandomWalk(), nsteps = 1, nchains = 2),
    )

    @test result isa NamedTuple
    @test hasproperty(result, :result)
    samples = result.result
    @test length(samples) > 0
    @test all(isfinite, samples.logd)

    sampled_params = Set(propertynames(samples.v))
    @test issubset((:mu, :sigma, :gaussian_A, :constPoly_C), sampled_params)

    mean_samples = mean(samples)
    mean_params = ModelParams(mean_samples)

    @test mean_params isa ModelParams

end
