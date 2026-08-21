using BAT: PosteriorMeasure
using GammaPeakFits:
    gaussian,
    compton,
    exGaussian,
    quad_polynomial,
    lin_polynomial,
    const_polynomial,
    peak_model,
    background_model,
    full_model,
    gaussian_integral,
    compton_integral,
    exGaussian_integral,
    quadPoly_integral,
    linPoly_integral,
    constPoly_integral,
    numerical_integral,
    midpoint_integral,
    analytical_integral

@testset "type stability" begin

    MU = 2048.0
    SIGMA = 10.0
    A = 1000.0
    H = 50.0
    TAU = 5.0
    C_QUAD = 1.0
    C_LIN = 10.0
    C_CONST = 100.0
    BIN_SIZE = 0.5
    WINDOW = 100.0

    model = ModelParams(
        peak = PeakParams(gaussian = GaussianParams(A = A, mu = MU, sigma = SIGMA)),
    )
    data = SpectrumData((MU-WINDOW/2), (MU+WINDOW/2), 1.0, model)
    X_ARRAY = data.bin_centers

    gaussian_params = GaussianParams(A = A, mu = MU, sigma = SIGMA)
    compton_params = ComptonParams(h = H, mu = MU, sigma = SIGMA)
    lowEnergyTail_params =
        ExGaussianParams(A = A, tau = TAU, is_lowEnergyTail = true, mu = MU, sigma = SIGMA)
    highEnergyTail_params =
        ExGaussianParams(A = A, tau = TAU, is_lowEnergyTail = false, mu = MU, sigma = SIGMA)
    quadPoly_params = QuadPolyParams(C = C_QUAD, mu = MU)
    linPoly_params = LinPolyParams(C = C_LIN, mu = MU)
    constPoly_params = ConstPolyParams(C = C_CONST)
    peak_params = PeakParams(
        gaussian = gaussian_params,
        compton = compton_params,
        lowEnergyTail = lowEnergyTail_params,
        highEnergyTail = highEnergyTail_params,
    )
    background_params = BackgroundParams(
        quadPoly = quadPoly_params,
        linPoly = linPoly_params,
        constPoly = constPoly_params,
    )
    model_params = ModelParams(peak = peak_params, background = background_params)

    @testset "component models (scalar/vector)" begin
        @test @inferred(gaussian(MU, gaussian_params)) isa Float64
        @test @inferred(gaussian(X_ARRAY, gaussian_params)) isa Vector{Float64}
        @test @inferred(compton(MU, compton_params)) isa Float64
        @test @inferred(compton(X_ARRAY, compton_params)) isa Vector{Float64}
        @test @inferred(exGaussian(MU, lowEnergyTail_params)) isa Float64
        @test @inferred(exGaussian(X_ARRAY, lowEnergyTail_params)) isa Vector{Float64}
        @test @inferred(exGaussian(MU, highEnergyTail_params)) isa Float64
        @test @inferred(exGaussian(X_ARRAY, highEnergyTail_params)) isa Vector{Float64}
        @test @inferred(quad_polynomial(MU, quadPoly_params)) isa Float64
        @test @inferred(quad_polynomial(X_ARRAY, quadPoly_params)) isa Vector{Float64}
        @test @inferred(lin_polynomial(MU, linPoly_params)) isa Float64
        @test @inferred(lin_polynomial(X_ARRAY, linPoly_params)) isa Vector{Float64}
        @test @inferred(const_polynomial(MU, constPoly_params)) isa Float64
        @test @inferred(const_polynomial(X_ARRAY, constPoly_params)) isa Vector{Float64}
    end

    @testset "combined models" begin
        @test @inferred(peak_model(MU, peak_params)) isa Float64
        @test @inferred(peak_model(X_ARRAY, peak_params)) isa Vector{Float64}
        @test @inferred(background_model(MU, background_params)) isa Float64
        @test @inferred(background_model(X_ARRAY, background_params)) isa Vector{Float64}
        @test @inferred(full_model(MU, model_params)) isa Float64
        @test @inferred(full_model(X_ARRAY, model_params)) isa Vector{Float64}
    end

    @testset "disabled sentinels" begin
        @test @inferred(peak_model(MU, PeakParams())) isa Float64
        @test @inferred(peak_model(X_ARRAY, PeakParams())) isa Vector{Float64}
        @test @inferred(background_model(MU, BackgroundParams())) isa Float64
        @test @inferred(background_model(X_ARRAY, BackgroundParams())) isa Vector{Float64}
        @test @inferred(full_model(MU, ModelParams())) isa Float64
        @test @inferred(full_model(X_ARRAY, ModelParams())) isa Vector{Float64}
    end

    @testset "analytical component integrals" begin
        @test @inferred(gaussian_integral(data, gaussian_params)) isa Vector{Float64}
        @test @inferred(compton_integral(data, compton_params)) isa Vector{Float64}
        @test @inferred(exGaussian_integral(data, lowEnergyTail_params)) isa Vector{Float64}
        @test @inferred(exGaussian_integral(data, highEnergyTail_params)) isa
              Vector{Float64}
        @test @inferred(quadPoly_integral(data, quadPoly_params)) isa Vector{Float64}
        @test @inferred(linPoly_integral(data, linPoly_params)) isa Vector{Float64}
        @test @inferred(constPoly_integral(data, constPoly_params)) isa Vector{Float64}
    end

    @testset "combined integrals" begin
        @test @inferred(midpoint_integral(data, model_params)) isa Vector{Float64}
        @test @inferred(analytical_integral(data, model_params)) isa Vector{Float64}
        @test @inferred(numerical_integral(data, model_params)) isa Vector{Float64}
    end

    @testset "disabled integrals" begin
        @test @inferred(analytical_integral(data, ModelParams())) isa Vector{Float64}
        @test @inferred(midpoint_integral(data, ModelParams())) isa Vector{Float64}
        @test @inferred(numerical_integral(data, ModelParams())) isa Vector{Float64}
    end

    @testset "poisson_ll" begin
        for integration_method in (Analytical(), Numerical(), Midpoint())
            configs =
                FitConfigs(mu = MU, sigma = SIGMA, integration_method = integration_method)
            @test @inferred(poisson_ll(data, model_params, configs)) isa Float64
        end
    end

    @testset "build_posterior" begin
        configs = FitConfigs(mu = MU, sigma = SIGMA)
        prior = build_prior(data, model_params, configs)
        @test @inferred(build_posterior(data, prior, configs)) isa PosteriorMeasure
    end

end
