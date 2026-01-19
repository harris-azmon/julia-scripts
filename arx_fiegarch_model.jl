using Turing
using Distributions
using LinearAlgebra
using Random
using StatsBase

# Import fractional differencing utilities from FIGARCH module
include("figarch_model.jl")
using .Main: fractional_diff_weights

"""
    ARX_FIEGARCH(returns, exog, ar_order, log_vol_init, n_lags)

ARX(p)-FIEGARCH(1,d,1) model combining:
- ARX mean equation with autoregressive lags and exogenous variables
- FIEGARCH volatility with fractional integration and asymmetric EGARCH effects

# Model specification
- Mean: r_t = φ_0 + φ_1*r_{t-1} + ... + φ_p*r_{t-p} + β_1*x_{1,t} + ... + β_k*x_{k,t} + ε_t
- Innovations: ε_t = σ_t * z_t, where z_t ~ N(0,1)
- Log-volatility: log(h_t) = ω + β_vol*log(h_{t-1}) + (φ_vol - β_vol)*g(z_{t-1}) + (1 - φ_vol)*λ_d(L)[g(z)]
- News function: g(z) = θ*z + γ*(|z| - sqrt(2/π))
- Fractional differencing: λ_d(L) = (1-L)^d applied to news history

# Arguments
- `returns`: Vector of return observations (length T)
- `exog`: Matrix of exogenous variables (T × k), or nothing for pure AR model
- `ar_order`: AR order p (number of lags in mean equation)
- `log_vol_init`: Initial log-volatility (can be missing for estimation)
- `n_lags`: Number of lags for fractional differencing approximation (default: 50)

# Parameters
## Mean equation
- φ_0: AR intercept
- φ: Vector of AR coefficients [φ_1, ..., φ_p]
- β_exog: Vector of exogenous variable coefficients [β_1, ..., β_k] (if exog provided)

## Volatility equation
- ω: Volatility intercept (unconstrained in log-space)
- d: Fractional integration parameter (0 < d < 1) - controls long memory
- φ_vol: Short-term volatility AR coefficient
- β_vol: Volatility persistence coefficient
- θ: Asymmetry/leverage parameter (negative → volatility increases after price drops)
- γ: Magnitude effect parameter (positive)

# Stationarity
- AR stationarity: Roots of characteristic polynomial outside unit circle
- FIEGARCH: log-volatility operates in log-space, no positivity constraint needed
- Long memory: 0 < d < 0.5 (stationary), 0.5 ≤ d < 1 (mean-reverting non-stationary)

# Example
```julia
# ARX(2)-FIEGARCH with 3 exogenous variables and d=0.3 long memory
exog = [interest_rates vix_index sentiment]  # T × 3 matrix
model = ARX_FIEGARCH(returns, exog, 2, missing, 50)
```
"""
@model function ARX_FIEGARCH(returns, exog=nothing, ar_order::Int=1,
                              log_vol_init=missing, n_lags::Int=50)
    T = length(returns)
    p = ar_order
    sqrt_2_pi = sqrt(2/π)  # E[|z|] for Normal distribution

    # Determine if exogenous variables are provided
    has_exog = !isnothing(exog)
    k = has_exog ? size(exog, 2) : 0

    # ===== MEAN EQUATION PARAMETERS =====

    # AR intercept
    φ_0 ~ Normal(0, 0.1)

    # AR coefficients with stationarity-inducing priors
    φ = Vector{typeof(φ_0)}(undef, p)
    for i in 1:p
        φ[i] ~ truncated(Normal(0, 0.3 / i), -0.99, 0.99)  # Decay with lag
    end

    # Exogenous variable coefficients
    if has_exog
        β_exog = Vector{typeof(φ_0)}(undef, k)
        for j in 1:k
            β_exog[j] ~ Normal(0, 1.0)  # Uninformative prior
        end
    end

    # ===== VOLATILITY EQUATION PARAMETERS =====

    # Intercept ω: unconstrained in log-volatility space
    ω ~ Normal(-0.1, 1.0)

    # d: Fractional integration parameter (long memory)
    # 0 < d < 0.5: stationary long memory
    # 0.5 ≤ d < 1: non-stationary but mean-reverting
    d_raw ~ Beta(2, 2)
    d = d_raw * 0.99 + 0.005  # Maps to (0.005, 0.995)

    # φ_vol: Short-term volatility dynamics (ARCH-like)
    φ_vol ~ truncated(Normal(0.3, 0.2), -0.99, 0.99)

    # β_vol: Volatility persistence (GARCH-like)
    β_vol ~ truncated(Normal(0.5, 0.2), -0.99, 0.99)

    # θ: Asymmetry/leverage effect
    # Negative θ → higher volatility after negative returns (leverage effect)
    θ ~ truncated(Normal(-0.1, 0.2), -1, 0.5)

    # γ: Magnitude effect (must be positive)
    # Log-normal reparametrization ensures γ > 0
    log_γ ~ Normal(log(0.5), 0.5)
    γ = exp(log_γ)

    # ===== INITIALIZATION =====

    # Initial log-volatility
    if ismissing(log_vol_init)
        log_vol_0 ~ Normal(log(var(returns)), 0.5)
    else
        log_vol_0 = log_vol_init
    end

    # Precompute fractional differencing weights for (1-L)^d
    fd_weights = fractional_diff_weights(d, n_lags)

    # Storage for history
    log_h = Vector{typeof(φ_0)}(undef, T)  # Log conditional variance
    news = Vector{typeof(φ_0)}(undef, T)   # EGARCH news term g(z_t)

    # Initialize first observation
    log_h[1] = log_vol_0

    # ===== TIME SERIES LOOP =====

    for t in 1:T
        # --- A. Compute ARX Mean ---
        if t <= p
            # For initial observations, use intercept only
            μ_t = φ_0
        else
            # AR component: φ_0 + φ_1*r_{t-1} + ... + φ_p*r_{t-p}
            μ_t = φ_0
            for i in 1:p
                μ_t += φ[i] * returns[t - i]
            end
        end

        # Exogenous component: β_1*x_{1,t} + ... + β_k*x_{k,t}
        if has_exog
            for j in 1:k
                μ_t += β_exog[j] * exog[t, j]
            end
        end

        # --- B. Current Volatility ---
        σ_t = exp(0.5 * log_h[t])

        # --- C. Observe Return ---
        returns[t] ~ Normal(μ_t, σ_t)

        # --- D. Compute Standardized Innovation ---
        z_t = (returns[t] - μ_t) / σ_t

        # --- E. Compute News Function g(z_t) ---
        news[t] = θ * z_t + γ * (abs(z_t) - sqrt_2_pi)

        # --- F. Update Volatility for Next Period ---
        if t < T
            # Compute fractional differencing term on news history
            # λ_d(L)[g(z)] = Σ_{k=1}^{min(t, n_lags)} π_k * g(z_{t-k+1})
            fd_term = 0.0
            limit = min(t, n_lags)
            for k in 1:limit
                fd_term += fd_weights[k + 1] * news[t - k + 1]
            end

            # FIEGARCH update equation
            # log(h_{t+1}) = ω + β_vol*log(h_t) + (φ_vol - β_vol)*g(z_t)
            #                + (1 - φ_vol)*fd_term
            log_h[t + 1] = ω + β_vol * log_h[t] + (φ_vol - β_vol) * news[t] +
                          (1 - φ_vol) * fd_term

            # Numerical stability: prevent extreme values
            # log(h) ∈ [-20, 10] => σ ∈ [exp(-10), exp(5)] ≈ [0.000045, 148]
            log_h[t + 1] = min(max(log_h[t + 1], -20.0), 10.0)
        end
    end
end

"""
    generate_arx_fiegarch_data(T, φ_0, φ, β_exog, exog_data, ω, d, φ_vol, β_vol, θ, γ;
                                log_vol_0=-2.0, n_lags=50, seed=123)

Generate synthetic data from an ARX-FIEGARCH model.

# Arguments
- `T`: Number of observations
- `φ_0`: AR intercept
- `φ`: Vector of AR coefficients (length p)
- `β_exog`: Vector of exogenous coefficients (length k), or nothing for pure AR
- `exog_data`: Matrix of exogenous variables (T × k), or nothing
- `ω`: Volatility intercept
- `d`: Fractional integration parameter (0 < d < 1)
- `φ_vol`: Short-term volatility AR coefficient
- `β_vol`: Volatility persistence
- `θ`: Asymmetry parameter
- `γ`: Magnitude parameter (> 0)
- `log_vol_0`: Initial log-volatility (default: -2.0)
- `n_lags`: Number of lags for fractional differencing (default: 50)
- `seed`: Random seed (default: 123)

# Returns
- NamedTuple with fields: returns, volatilities, log_volatilities, innovations, news

# Example
```julia
# ARX(1)-FIEGARCH with one exogenous variable
exog_data = randn(1000, 1)
data = generate_arx_fiegarch_data(1000, 0.0, [0.1], [0.5], exog_data,
                                   -0.1, 0.3, 0.3, 0.5, -0.1, 0.5)
```
"""
function generate_arx_fiegarch_data(T::Int, φ_0, φ::Vector, β_exog, exog_data,
                                     ω, d, φ_vol, β_vol, θ, γ;
                                     log_vol_0=-2.0, n_lags=50, seed=123)
    Random.seed!(seed)

    p = length(φ)
    has_exog = !isnothing(β_exog)
    k = has_exog ? length(β_exog) : 0
    sqrt_2_pi = sqrt(2/π)

    # Precompute fractional differencing weights
    fd_weights = fractional_diff_weights(d, n_lags)

    # Initialize storage
    returns = zeros(T)
    log_h = zeros(T)
    volatilities = zeros(T)
    innovations = randn(T)
    news = zeros(T)

    # First observation
    log_h[1] = log_vol_0
    volatilities[1] = exp(0.5 * log_h[1])

    # Compute mean for first observation
    μ_1 = φ_0
    if has_exog
        for j in 1:k
            μ_1 += β_exog[j] * exog_data[1, j]
        end
    end

    returns[1] = μ_1 + volatilities[1] * innovations[1]
    z_1 = innovations[1]
    news[1] = θ * z_1 + γ * (abs(z_1) - sqrt_2_pi)

    # Generate time series
    for t in 2:T
        # Compute ARX mean
        if t <= p
            μ_t = φ_0
        else
            μ_t = φ_0
            for i in 1:p
                μ_t += φ[i] * returns[t - i]
            end
        end

        # Add exogenous effects
        if has_exog
            for j in 1:k
                μ_t += β_exog[j] * exog_data[t, j]
            end
        end

        # Compute fractional differencing term
        fd_term = 0.0
        limit = min(t - 1, n_lags)
        for k_lag in 1:limit
            fd_term += fd_weights[k_lag + 1] * news[t - k_lag]
        end

        # FIEGARCH update
        log_h[t] = ω + β_vol * log_h[t-1] + (φ_vol - β_vol) * news[t-1] +
                   (1 - φ_vol) * fd_term
        log_h[t] = min(max(log_h[t], -20.0), 10.0)  # Numerical stability

        # Generate return
        volatilities[t] = exp(0.5 * log_h[t])
        returns[t] = μ_t + volatilities[t] * innovations[t]

        # Compute news for next iteration
        z_t = innovations[t]
        news[t] = θ * z_t + γ * (abs(z_t) - sqrt_2_pi)
    end

    return (
        returns = returns,
        volatilities = volatilities,
        log_volatilities = log_h,
        innovations = innovations,
        news = news
    )
end

"""
    fit_arx_fiegarch(returns; exog=nothing, ar_order=1, n_samples=1000, n_chains=4,
                     n_lags=50, log_vol_init=missing)

Fit ARX-FIEGARCH model to returns data using NUTS sampler.

# Arguments
- `returns`: Vector of return observations
- `exog`: Matrix of exogenous variables (T × k), or nothing for pure AR-FIEGARCH
- `ar_order`: AR order p in mean equation (default: 1)
- `n_samples`: Number of posterior samples per chain (default: 1000)
- `n_chains`: Number of MCMC chains (default: 4)
- `n_lags`: Number of lags for fractional differencing (default: 50)
- `log_vol_init`: Initial log-volatility (missing for estimation)

# Returns
- Chains object with posterior samples

# Example
```julia
# Fit ARX(1)-FIEGARCH with exogenous variables
exog = [interest_rates vix_index]
chain = fit_arx_fiegarch(returns, exog=exog, ar_order=1, n_samples=1000)

# Extract fractional integration parameter
d_samples = chain[:d]
mean(d_samples)  # Posterior mean of long memory parameter
```
"""
function fit_arx_fiegarch(returns;
                          exog=nothing,
                          ar_order=1,
                          n_samples=1000,
                          n_chains=4,
                          n_lags=50,
                          log_vol_init=missing)

    println("Fitting ARX($ar_order)-FIEGARCH(1,d,1) model")
    println("  Total observations: $(length(returns))")

    if !isnothing(exog)
        k = size(exog, 2)
        println("  Exogenous variables: $k")
    else
        println("  Exogenous variables: none (pure AR-FIEGARCH)")
    end

    println("  AR order: $ar_order")
    println("  Fractional diff lags: $n_lags")

    model = ARX_FIEGARCH(returns, exog, ar_order, log_vol_init, n_lags)

    # Sample using NUTS
    println("\nStarting NUTS sampling...")

    # Use MCMCThreads() if multiple threads available
    sampler_type = Threads.nthreads() > 1 ? MCMCThreads() : MCMCSerial()

    chain = sample(model, NUTS(0.65), sampler_type, n_samples, n_chains,
                   progress=true)

    return chain
end

# Export main functions
export ARX_FIEGARCH, generate_arx_fiegarch_data, fit_arx_fiegarch

println("ARX-FIEGARCH module loaded successfully!")
println("Main functions:")
println("  - generate_arx_fiegarch_data(): Generate synthetic ARX-FIEGARCH data")
println("  - fit_arx_fiegarch(): Fit ARX(p)-FIEGARCH(1,d,1) model")
println()
println("Model: ARX(p)-FIEGARCH(1,d,1)")
println("  • Combines autoregressive + exogenous mean with FIEGARCH volatility")
println("  • Long memory in volatility via fractional integration (d parameter)")
println("  • Asymmetric volatility response via EGARCH news function")
println("  • Flexible mean specification: AR lags + exogenous variables")
println()
println("Key parameters:")
println("  • d ∈ (0,1): Fractional integration (long memory strength)")
println("  • θ < 0: Leverage effect (asymmetry)")
println("  • γ > 0: Magnitude effect")
println("  • φ_vol, β_vol: Short-term volatility dynamics")
println()
println("Features:")
println("  ✓ Log-volatility space (no positivity constraints)")
println("  ✓ Intelligent reparametrization (Beta for d, log-normal for γ)")
println("  ✓ Supports pure AR-FIEGARCH (exog=nothing)")
println("  ✓ Numerical stability safeguards")
println("  ✓ Automatic threading detection")
println("  ✓ Efficient fractional differencing")
println()
