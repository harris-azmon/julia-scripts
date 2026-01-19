# Julia Scripts

This repository contains Julia scripts for statistical modeling and time series analysis.

## Contents

- **`egarch_model.jl`** - EGARCH(1,1) model implementation using Turing.jl
- **`simple_test.jl`** - Simple test demonstrating EGARCH functionality
- **`test_egarch_full.jl`** - Comprehensive test with MCMC sampling
- **`validate_syntax.jl`** - Syntax validation utility
- **`INSTALLATION.md`** - Detailed installation notes and segfault fix documentation

## Quick Start

```bash
# Set up Julia path
export PATH="$HOME/.juliaup/bin:$PATH"

# Run the simple test
julia simple_test.jl

# Or use the EGARCH model interactively
julia
```

```julia
include("egarch_model.jl")

# Generate synthetic data
data = generate_egarch_data(200, 0.0, -0.3, 0.15, 0.85, -0.05, 0.5)

# Fit model (requires tuning for your data)
chain = fit_egarch(data.returns, n_samples=1000, n_chains=4)
```

## Installed Packages

- **Turing** (v0.42.4) - Probabilistic programming and Bayesian inference
- **Distributions** - Probability distributions
- **StatsBase** - Basic statistical functions
- **LinearAlgebra** (stdlib) - Linear algebra operations
- **Random** (stdlib) - Random number generation

**Important:** See `INSTALLATION.md` for package installation details and segmentation fault fix.

## Julia Installation

Julia has been installed using **juliaup**, the official Julia installer and version manager.

### Installed Versions
- Julia: 1.12.4
- Juliaup: 1.18.9

### Using Julia

To use Julia in this environment, ensure the PATH is set:

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
julia
```

Or simply reload your shell configuration:

```bash
source ~/.bashrc
```

### Managing Julia Versions with Juliaup

Juliaup makes it easy to manage Julia versions:

```bash
# Update Julia
juliaup update

# List available channels
juliaup list

# Install specific version
juliaup add 1.10

# Set default version
juliaup default 1.10

# Show current status
juliaup status
```

### Uninstalling

To uninstall Julia and juliaup:

```bash
juliaup self uninstall
```

## Getting Started

Create Julia scripts with `.jl` extension and run them:

```bash
julia script.jl
```

Or use the Julia REPL interactively:

```bash
julia
```
