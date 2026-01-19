# Julia Scripts

This repository contains Julia scripts and programs.

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
