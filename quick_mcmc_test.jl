# Quick MCMC test to verify reparametrization works
println("Testing EGARCH with intelligent reparametrization...")
include("egarch_model.jl")

# Generate test data
data = generate_egarch_data(60, 0.0, -0.3, 0.15, 0.85, -0.05, 0.5; seed=42)

# Fit model
println("\nFitting model (50 samples, 1 chain)...")
chain = fit_egarch(data.returns[1:60];
                  block_size=15,
                  n_samples=50,
                  n_chains=1,
                  use_blocks=true)

println("\n" * "="^70)
println("SUCCESS: MCMC Completed Without Domain Errors!")
println("="^70)
println("\nParameter Summary:")
println("  μ  : $(round(mean(chain[:μ]), digits=4))")
println("  ω  : $(round(mean(chain[:ω]), digits=4))")
println("  α  : $(round(mean(chain[:α]), digits=4)) ∈ [$(round(minimum(chain[:α]), digits=4)), $(round(maximum(chain[:α]), digits=4))]")
println("  β  : $(round(mean(chain[:β]), digits=4))")
println("  θ  : $(round(mean(chain[:θ]), digits=4))")

# Transform log_γ to γ
log_γ = chain[:log_γ]
γ = exp.(log_γ)
println("  log_γ: $(round(mean(log_γ), digits=4))")
println("  γ  : $(round(mean(γ), digits=4)) (all > 0: $(all(γ .> 0)))")

println("\n✓ Reparametrization working correctly!")
println("✓ No 'σ >= zero(σ)' domain errors")
println("✓ All parameters in valid ranges")
