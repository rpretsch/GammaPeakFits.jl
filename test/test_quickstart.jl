using BAT: TransformedMCMC, bat_sample, RandomWalk
using Random: seed!
using StatsBase: mean

@testset "Quick start example (README / module docstring)" begin

    seed!(42)

    # Expected peak
    mu = 2048.0     # keV
    sigma = 5.0     # keV

    # Generate data
    peak_counts = 1000.0          # counts
    background = 100.0            # counts/keV
    energy_range = (0.0, 4096.0)  # keV, outer bin edges
    bin_size = 0.5                # keV

    truth = peak_counts * GaussPeak(mu, sigma) + PolynomialShape(background)
    data = sample_spectrum(truth; bin_size, energy_range)

    # cut appropriate fit window
    fit_data = cut_data(data; center = mu, width = 10 * sigma)

    # Model kernel: parameters -> spectral model
    model = @pf $n * GaussPeak($mu, $sigma) + PolynomialShape($b)

    # Data-driven default priors, the prior takes the entries the model uses
    defaults =
        merge(peak_priors(fit_data; mu, sigma), background_priors(fit_data; mu, sigma))
    prior = build_prior(model, defaults)
    posterior = build_posterior(fit_data, model, prior)

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
    @test propertynames(samples.v) == (:n, :mu, :sigma, :b)

    # Expected counts of the fitted model, e.g. for plotting
    mean_params = mean(samples)
    fitted = expected_counts(model(mean_params), fit_data)
    @test length(fitted) == length(fit_data.weights)

end
