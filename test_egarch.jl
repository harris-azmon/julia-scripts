# Simple test script for EGARCH model
# This tests basic functionality without running full MCMC sampling

println("Testing EGARCH model script...")
println()

# Try to include the main script
try
    include("egarch_model.jl")
    println("✓ Script loaded successfully!")
    println()

    # Test helper functions
    println("Testing helper functions:")

    # Test asymmetry function
    z_test = 0.5
    θ_test = -0.1
    γ_test = 0.8
    g_z = asymmetry_function(z_test, θ_test, γ_test)
    println("  ✓ asymmetry_function($(z_test), $(θ_test), $(γ_test)) = $(g_z)")

    # Test update_log_volatility
    log_vol_prev = -2.0
    z_prev = 0.3
    ω = -0.5
    α = 0.2
    β = 0.9
    new_log_vol = update_log_volatility(log_vol_prev, z_prev, ω, α, β, θ_test, γ_test)
    println("  ✓ update_log_volatility() = $(new_log_vol)")

    println()
    println("Testing data generation:")

    # Test generate_egarch_data
    T = 100
    μ = 0.001
    data = generate_egarch_data(T, μ, ω, α, β, θ_test, γ_test; seed=42)
    println("  ✓ Generated $(T) observations")
    println("  ✓ Mean return: $(round(mean(data.returns), digits=4))")
    println("  ✓ Mean volatility: $(round(mean(data.volatilities), digits=4))")
    println("  ✓ Volatility range: [$(round(minimum(data.volatilities), digits=4)), $(round(maximum(data.volatilities), digits=4))]")

    println()
    println("All basic tests passed! ✓")
    println()
    println("Note: To run full MCMC sampling, ensure all packages are properly installed:")
    println("  julia> using Pkg")
    println("  julia> Pkg.add([\"Turing\", \"Distributions\", \"StatsBase\"])")

catch e
    println("Error loading script:")
    println(e)
    exit(1)
end
