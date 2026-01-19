using Turing
using Distributions
using LinearAlgebra
using Random
using StatsBase

"""
    asymmetry_function(z, θ, γ)

Compute the asymmetry function g(z) for EGARCH model.
g(z) = θ*z + γ*(|z| - E[|z|])
where E[|z|] = sqrt(2/π) for standard normal
"""
function asymmetry_function(z, θ, γ)
    expected_abs_z = sqrt(2/π)
    return θ * z + γ * (abs(z) - expected_abs_z)
end

"""
    update_log_volatility(log_vol_prev, z_prev, ω, α, β, θ, γ)

Update log-volatility using EGARCH(1,1) dynamics:
log(σ²_t) = ω + α * g(z_{t-1}) + β * log(σ²_{t-1})
"""
function update_log_volatility(log_vol_prev, z_prev, ω, α, β, θ, γ)
    g_z = asymmetry_function(z_prev, θ, γ)
    return ω + α * g_z + β * log_vol_prev
end

"""
    EGARCH_block(returns, block_size, n_blocks, log_vol_init)

EGARCH(1,1) model with block-wise processing using scan.

# Arguments
- `returns`: Vector of return observations
- `block_size`: Number of observations per block
- `n_blocks`: Number of blocks to process
- `log_vol_init`: Initial log-volatility (can be missing for estimation)

# Model specification
- Returns: r_t = μ + ε_t
- Innovations: ε_t = σ_t * z_t, where z_t ~ N(0,1)
- Log-volatility: log(σ²_t) = ω + α * g(z_{t-1}) + β * log(σ²_{t-1})
- Asymmetry: g(z) = θ*z + γ*(|z| - sqrt(2/π))

# Parameters
- μ: mean return
- ω: intercept in log-volatility
- α: ARCH coefficient
- β: GARCH coefficient (persistence)
- θ: leverage effect (typically < 0)
- γ: magnitude effect (typically > 0)
"""
@model function EGARCH_block(returns, block_size, n_blocks, log_vol_init=missing)
    T = length(returns)

    # Priors for EGARCH parameters
    μ ~ Normal(0, 0.1)
    ω ~ Normal(-5, 2)
    α ~ Normal(0, 0.5)
    β ~ Beta(20, 2)  # Encourage high persistence (mean ≈ 0.91)
    θ ~ Normal(-0.1, 0.3)  # Leverage effect, typically negative
    γ ~ truncated(Normal(0.8, 0.5), 0.01, Inf)  # Magnitude effect

    # Initial log-volatility
    if ismissing(log_vol_init)
        log_vol_0 ~ Normal(-2, 1)  # Prior for initial log-volatility
    else
        log_vol_0 = log_vol_init
    end

    # Storage for standardized residuals
    z = Vector{Float64}(undef, T)
    log_vols = Vector{Float64}(undef, T)

    # Initial values
    z[1] = (returns[1] - μ) / exp(log_vol_0/2)
    log_vols[1] = log_vol_0

    # Process first observation
    returns[1] ~ Normal(μ, exp(log_vol_0/2))

    # Scan through blocks
    current_log_vol = log_vol_0

    for block in 1:n_blocks
        start_idx = (block - 1) * block_size + 2  # +2 because we start from t=2
        end_idx = min(block * block_size + 1, T)

        if start_idx > T
            break
        end

        # Process each observation in the block
        for t in start_idx:end_idx
            # Update log-volatility based on previous standardized residual
            current_log_vol = update_log_volatility(
                current_log_vol, z[t-1], ω, α, β, θ, γ
            )
            log_vols[t] = current_log_vol

            # Current volatility
            σ_t = exp(current_log_vol / 2)

            # Observe return
            returns[t] ~ Normal(μ, σ_t)

            # Compute standardized residual
            z[t] = (returns[t] - μ) / σ_t
        end
    end

    # Return useful quantities for diagnostics
    return (z=z, log_vols=log_vols, volatilities=exp.(log_vols ./ 2))
end

"""
    EGARCH_full(returns, log_vol_init)

EGARCH(1,1) model processing the full time series at once.
This is simpler but less memory efficient for large datasets.
"""
@model function EGARCH_full(returns, log_vol_init=missing)
    T = length(returns)

    # Priors
    μ ~ Normal(0, 0.1)
    ω ~ Normal(-5, 2)
    α ~ Normal(0, 0.5)
    β ~ Beta(20, 2)
    θ ~ Normal(-0.1, 0.3)
    γ ~ truncated(Normal(0.8, 0.5), 0.01, Inf)

    # Initial log-volatility
    if ismissing(log_vol_init)
        log_vol_0 ~ Normal(-2, 1)
    else
        log_vol_0 = log_vol_init
    end

    # Process time series
    log_vol = log_vol_0

    for t in 1:T
        σ_t = exp(log_vol / 2)
        returns[t] ~ Normal(μ, σ_t)

        if t < T
            z_t = (returns[t] - μ) / σ_t
            log_vol = update_log_volatility(log_vol, z_t, ω, α, β, θ, γ)
        end
    end
end

"""
    generate_egarch_data(T, μ, ω, α, β, θ, γ; log_vol_0=-2.0, seed=123)

Generate synthetic data from an EGARCH(1,1) model.

# Returns
- NamedTuple with fields: returns, volatilities, log_volatilities, innovations
"""
function generate_egarch_data(T::Int, μ, ω, α, β, θ, γ;
                             log_vol_0=-2.0, seed=123)
    Random.seed!(seed)

    returns = zeros(T)
    log_vols = zeros(T)
    volatilities = zeros(T)
    innovations = randn(T)

    log_vols[1] = log_vol_0
    volatilities[1] = exp(log_vol_0 / 2)
    returns[1] = μ + volatilities[1] * innovations[1]

    for t in 2:T
        z_prev = innovations[t-1]
        log_vols[t] = update_log_volatility(log_vols[t-1], z_prev, ω, α, β, θ, γ)
        volatilities[t] = exp(log_vols[t] / 2)
        returns[t] = μ + volatilities[t] * innovations[t]
    end

    return (
        returns = returns,
        volatilities = volatilities,
        log_volatilities = log_vols,
        innovations = innovations
    )
end

"""
    fit_egarch(returns; block_size=50, n_samples=1000, n_chains=4,
               log_vol_init=missing, use_blocks=true)

Fit EGARCH model to returns data using NUTS sampler.

# Arguments
- `returns`: Vector of return observations
- `block_size`: Size of each block for processing
- `n_samples`: Number of posterior samples per chain
- `n_chains`: Number of MCMC chains
- `log_vol_init`: Initial log-volatility (missing for estimation)
- `use_blocks`: Whether to use block-wise processing

# Returns
- Chains object with posterior samples
"""
function fit_egarch(returns;
                    block_size=50,
                    n_samples=1000,
                    n_chains=4,
                    log_vol_init=missing,
                    use_blocks=true)

    if use_blocks
        T = length(returns)
        n_blocks = ceil(Int, (T - 1) / block_size)

        println("Fitting EGARCH model with block-wise processing")
        println("  Total observations: $T")
        println("  Block size: $block_size")
        println("  Number of blocks: $n_blocks")

        model = EGARCH_block(returns, block_size, n_blocks, log_vol_init)
    else
        println("Fitting EGARCH model on full time series")
        model = EGARCH_full(returns, log_vol_init)
    end

    # Sample using NUTS
    println("\nStarting NUTS sampling...")

    # Use MCMCThreads() if multiple threads available, otherwise MCMCSerial()
    sampler_type = Threads.nthreads() > 1 ? MCMCThreads() : MCMCSerial()

    chain = sample(model, NUTS(0.65), sampler_type, n_samples, n_chains,
                   progress=true)

    return chain
end

"""
    forecast_volatility(chain, returns, h=10; block_size=50)

Forecast volatility h steps ahead using posterior samples.

# Arguments
- `chain`: Posterior samples from fit_egarch
- `returns`: Historical return observations
- `h`: Forecast horizon
- `block_size`: Block size used in fitting

# Returns
- Matrix of volatility forecasts (n_samples × h)
"""
function forecast_volatility(chain, returns, h=10; block_size=50)
    n_samples = length(chain)
    forecasts = zeros(n_samples, h)

    for i in 1:n_samples
        # Extract parameters
        params = chain[i]
        μ = params[:μ]
        ω = params[:ω]
        α = params[:α]
        β = params[:β]
        θ = params[:θ]
        γ = params[:γ]

        # Get final log-volatility from data
        T = length(returns)
        log_vol = haskey(params, :log_vol_0) ? params[:log_vol_0] : -2.0

        # Forward simulate through historical data to get final state
        for t in 1:T
            σ_t = exp(log_vol / 2)
            z_t = (returns[t] - μ) / σ_t

            if t < T
                log_vol = update_log_volatility(log_vol, z_t, ω, α, β, θ, γ)
            end
        end

        # Forecast
        z_last = (returns[end] - μ) / exp(log_vol / 2)

        for step in 1:h
            log_vol = update_log_volatility(log_vol, z_last, ω, α, β, θ, γ)
            forecasts[i, step] = exp(log_vol / 2)

            # For next step, assume z_t = 0 (expected value)
            z_last = 0.0
        end
    end

    return forecasts
end

# Export main functions
export EGARCH_block, EGARCH_full, generate_egarch_data
export fit_egarch, forecast_volatility
export asymmetry_function, update_log_volatility

println("EGARCH module loaded successfully!")
println("Main functions:")
println("  - generate_egarch_data(): Generate synthetic EGARCH data")
println("  - fit_egarch(): Fit EGARCH model with configurable block size")
println("  - forecast_volatility(): Forecast future volatility")
