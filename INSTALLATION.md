# Julia Installation and Package Setup

## Julia Installation

Julia was installed using **juliaup**, the official Julia installer and version manager.

```bash
curl -fsSL https://install.julialang.org | sh -s -- -y
```

**Installed versions:**
- Julia: 1.12.4
- Juliaup: 1.18.9

## Segmentation Fault Fix

### Problem
When installing Julia packages (particularly Turing with many dependencies), segmentation faults occurred during artifact installation:

```
[XXXX] signal 11 (1): Segmentation fault
```

The crash happened in the task scheduler during parallel artifact downloads.

### Root Cause
The segmentation fault was caused by concurrent artifact installation operations in environments with limited threading support or sandbox restrictions.

### Solution
Use single-threaded mode with disabled auto-precompilation during package installation:

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
JULIA_NUM_THREADS=1 julia -e 'using Pkg; ENV["JULIA_PKG_PRECOMPILE_AUTO"]=0; Pkg.add("PackageName"; io=devnull)'
```

**Key workarounds:**
1. `JULIA_NUM_THREADS=1` - Force single-threaded execution
2. `ENV["JULIA_PKG_PRECOMPILE_AUTO"]=0` - Disable automatic precompilation during installation
3. `io=devnull` - Suppress verbose output

### Packages Installed
The following packages were successfully installed using this method:

- **Distributions** - Probability distributions
- **StatsBase** - Basic statistics
- **Turing** (v0.42.4) - Probabilistic programming and MCMC sampling
  - Includes 238+ dependencies

## Usage After Installation

Once installed, packages work normally:

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
julia -e 'using Turing, Distributions, StatsBase'
```

Precompilation happens on first `using` statement and works without issues.

## Code Fixes for Single-Threaded Environments

The EGARCH model code was updated to automatically detect threading capability:

```julia
# Original (would fail in single-threaded environment)
chain = sample(model, NUTS(0.65), MCMCThreads(), n_samples, n_chains)

# Fixed (adapts to environment)
sampler_type = Threads.nthreads() > 1 ? MCMCThreads() : MCMCSerial()
chain = sample(model, NUTS(0.65), sampler_type, n_samples, n_chains)
```

## Environment Details

- **Platform:** Linux 4.4.0
- **Memory:** 21GB available
- **Julia Threads:** 1 (default)

## Recommendations

For production environments:
1. Use the installation workaround for initial package setup
2. Consider enabling multi-threading: `export JULIA_NUM_THREADS=4`
3. Ensure adequate memory for MCMC sampling (>2GB recommended)
4. Test package loading after installation to verify precompilation

## Testing

Run the test suite to verify installation:

```bash
julia simple_test.jl
```

This validates:
- Package loading
- Data generation
- Model definitions
- Basic functionality

For full MCMC testing (optional):
```bash
julia test_egarch_full.jl
```
