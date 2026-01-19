using Turing
using Distributions
using LinearAlgebra
using Random
using StatsBase

"""
    fractional_diff_weights(d, n_lags)

Compute fractional differencing weights for (1-L)^d using binomial expansion.

The weights are given by:
w_k = Γ(k - d) / (Γ(-d) * Γ(k + 1))

Or recursively:
w_0 = 1
w_k = w_{k-1} * (k - 1 - d) / k

# Arguments
- `d`: Fractional differencing parameter (0 < d < 1)
- `n_lags`: Number of lags to compute

# Returns
- Vector of fractional differencing weights [w_0, w_1, ..., w_{n_lags}]
"""
function fractional_diff_weights(d, n_lags::Int)
    weights = zeros(n_lags + 1)
    weights[1] = 1.0

    for k in 1:n_lags
        weights[k + 1] = weights[k] * (k - 1 - d) / k
    end

    return weights
end

"""
    apply_fractional_diff(x, d, n_lags)

Apply fractional differencing (1-L)^d to a series x.

# Arguments
- `x`: Time series vector
- `d`: Fractional differencing parameter
- `n_lags`: Number of lags to use in approximation

# Returns
- Fractionally differenced series
"""
function apply_fractional_diff(x, d, n_lags::Int)
    T = length(x)
    weights = fractional_diff_weights(d, n_lags)
    y = similar(x)

    for t in 1:T
        y[t] = 0.0
        for k in 0:min(t-1, n_lags)
            y[t] += weights[k + 1] * x[t - k]
        end
    end

    return y
end

"""
    FIGARCH(returns, log_vol_init)

FIGARCH(1,d,1) model with fractionally integrated volatility dynamics.

# Model specification
- Mean: r_t = μ + ε_t
- Innovations: ε_t = σ_t * z_t, where z_t ~ N(0,1)
- Volatility: σ²_t = ω + λ(L) ε²_t
- Lambda polynomial: λ(L) = [1 - β(L)]^(-1) [1 - β(L) - φ(L)(1-L)^d]

Simplified representation:
σ²_t = ω + β σ²_{t-1} + [φ - β + (1-φ)λ_d(L)] ε²_{t-1}

where λ_d(L) represents the fractional differencing operator.

# Parameters
- μ: mean return
- ω: volatility intercept (ω > 0)
- φ: AR coefficient in variance equation
- d: fractional differencing parameter (0 < d < 1, provides long memory)
- β: GARCH coefficient (persistence)

# Stationarity
- Requires: β + d/2 < 1 (approximate condition)
- d = 0: reduces to GARCH(1,1)
- d = 1: reduces to IGARCH (non-stationary)
- 0 < d < 1: long memory (hyperbolic decay)

# Arguments
- `returns`: Vector of return observations
- `log_vol_init`: Initial log-volatility (can be missing for estimation)
"""
@model function FIGARCH(returns, log_vol_init=missing, n_lags::Int=20)
    T = length(returns)

    # Priors with intelligent reparametrization
    μ ~ Normal(0, 0.1)

    # Volatility intercept: must be positive, use log-normal
    log_ω ~ Normal(log(0.01), 1.0)
    ω = exp(log_ω)  # Ensures ω > 0

    # φ: AR coefficient, typically positive
    φ ~ truncated(Normal(0.3, 0.2), 0.01, 0.99)

    # d: Fractional integration parameter (key parameter for long memory)
    # Use Beta distribution rescaled to (0,1) for proper support
    d_raw ~ Beta(2, 2)  # Centered around 0.5
    d = d_raw * 0.99 + 0.005  # Maps to (0.005, 0.995)

    # β: GARCH persistence
    # Constraint: β + d/2 < 1 for stationarity (approximate)
    # Use constrained prior
    β ~ Beta(10, 2)  # Encourages persistence

    # Initial log-volatility
    if ismissing(log_vol_init)
        log_vol_0 ~ Normal(log(var(returns)), 0.5)
    else
        log_vol_0 = log_vol_init
    end

    # Precompute fractional differencing weights
    fd_weights = fractional_diff_weights(d, n_lags)

    # Storage
    σ² = Vector{typeof(μ)}(undef, T)
    ε² = Vector{typeof(μ)}(undef, T)

    # Initialize
    σ²[1] = exp(log_vol_0)
    ε²[1] = (returns[1] - μ)^2

    # Observe first return
    returns[1] ~ Normal(μ, sqrt(max(σ²[1], 1e-6)))

    # Process time series
    for t in 2:T
        # Compute fractional differencing term
        fd_term = 0.0
        for k in 1:min(t-1, n_lags)
            fd_term += fd_weights[k + 1] * ε²[t - k]
        end

        # FIGARCH variance equation
        # σ²_t = ω + β σ²_{t-1} + (φ - β) ε²_{t-1} + (1-φ) * fd_term
        σ²[t] = ω + β * σ²[t-1] + (φ - β) * ε²[t-1] + (1 - φ) * fd_term

        # Ensure positivity (numerical safety)
        σ²[t] = max(σ²[t], 1e-6)

        # Observe return
        σ_t = sqrt(σ²[t])
        returns[t] ~ Normal(μ, σ_t)

        # Store squared innovation
        ε²[t] = (returns[t] - μ)^2
    end
end

"""
    generate_figarch_data(T, μ, ω, φ, d, β; log_vol_0=-2.0, n_lags=20, seed=123)

Generate synthetic data from a FIGARCH(1,d,1) model.

# Arguments
- `T`: Number of observations
- `μ`: Mean return
- `ω`: Volatility intercept (must be > 0)
- `φ`: AR coefficient in variance
- `d`: Fractional integration parameter (0 < d < 1)
- `β`: GARCH coefficient
- `log_vol_0`: Initial log-volatility (default: -2.0)
- `n_lags`: Number of lags for fractional differencing (default: 20)
- `seed`: Random seed (default: 123)

# Returns
- NamedTuple with fields: returns, volatilities, log_volatilities, innovations

# Example
```julia
# FIGARCH with d=0.4 (moderate long memory)
data = generate_figarch_data(1000, 0.0, 0.01, 0.3, 0.4, 0.6)
```
"""
function generate_figarch_data(T::Int, μ, ω, φ, d, β;
                               log_vol_0=-2.0, n_lags=20, seed=123)
    Random.seed!(seed)

    # Precompute fractional differencing weights
    fd_weights = fractional_diff_weights(d, n_lags)

    returns = zeros(T)
    σ² = zeros(T)
    volatilities = zeros(T)
    innovations = randn(T)
    ε² = zeros(T)

    # Initialize
    σ²[1] = exp(log_vol_0)
    volatilities[1] = sqrt(σ²[1])
    returns[1] = μ + volatilities[1] * innovations[1]
    ε²[1] = (returns[1] - μ)^2

    for t in 2:T
        # Compute fractional differencing term
        fd_term = 0.0
        for k in 1:min(t-1, n_lags)
            fd_term += fd_weights[k + 1] * ε²[t - k]
        end

        # FIGARCH variance equation
        σ²[t] = ω + β * σ²[t-1] + (φ - β) * ε²[t-1] + (1 - φ) * fd_term
        σ²[t] = max(σ²[t], 1e-8)  # Ensure positivity

        volatilities[t] = sqrt(σ²[t])
        returns[t] = μ + volatilities[t] * innovations[t]
        ε²[t] = (returns[t] - μ)^2
    end

    return (
        returns = returns,
        volatilities = volatilities,
        log_volatilities = log.(σ²),
        innovations = innovations
    )
end

"""
    fit_figarch(returns; n_samples=1000, n_chains=4, n_lags=20, log_vol_init=missing)

Fit FIGARCH(1,d,1) model to returns data using NUTS sampler.

# Arguments
- `returns`: Vector of return observations
- `n_samples`: Number of posterior samples per chain (default: 1000)
- `n_chains`: Number of MCMC chains (default: 4)
- `n_lags`: Number of lags for fractional differencing (default: 20)
- `log_vol_init`: Initial log-volatility (missing for estimation)

# Returns
- Chains object with posterior samples

# Example
```julia
chain = fit_figarch(returns, n_samples=1000, n_chains=4)

# Extract parameters
d_samples = chain[:d]  # Fractional integration parameter
```
"""
function fit_figarch(returns;
                     n_samples=1000,
                     n_chains=4,
                     n_lags=20,
                     log_vol_init=missing)

    println("Fitting FIGARCH(1,d,1) model")
    println("  Total observations: $(length(returns))")
    println("  Fractional diff lags: $n_lags")

    model = FIGARCH(returns, log_vol_init, n_lags)

    # Sample using NUTS
    println("\nStarting NUTS sampling...")

    # Use MCMCThreads() if multiple threads available, otherwise MCMCSerial()
    sampler_type = Threads.nthreads() > 1 ? MCMCThreads() : MCMCSerial()

    chain = sample(model, NUTS(0.65), sampler_type, n_samples, n_chains,
                   progress=true)

    return chain
end

# Export main functions
export FIGARCH, generate_figarch_data, fit_figarch
export fractional_diff_weights, apply_fractional_diff

println("FIGARCH module loaded successfully!")
println("Main functions:")
println("  - generate_figarch_data(): Generate synthetic FIGARCH data")
println("  - fit_figarch(): Fit FIGARCH(1,d,1) model")
println("  - fractional_diff_weights(): Compute fractional differencing weights")
println()
println("Model: FIGARCH(1,d,1)")
println("  • Long memory volatility via fractional integration")
println("  • Fractional parameter d ∈ (0,1) controls memory")
println("  • d → 0: Short memory (like GARCH)")
println("  • d → 1: Infinite memory (like IGARCH)")
println("  • 0 < d < 1: Hyperbolic decay (long memory)")
println()
println("Features:")
println("  ✓ Intelligent reparametrization (log-normal for ω > 0)")
println("  ✓ Beta distribution for d ∈ (0,1)")
println("  ✓ Numerical stability safeguards")
println("  ✓ Automatic threading detection")
println("  ✓ Efficient fractional differencing via binomial weights")
