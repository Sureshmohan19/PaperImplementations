# PaperImplementations

> **Miniforge Version:** 25.9.1  
> **Mamba Version:** 2.3.3

This repository contains concise implementations and notes from various machine learning and AI research papers.
Each folder includes runnable code, explanations, and insights extracted for future research use.

## Prerequisites

**Required:** Mamba (conda replacement, much faster)

Install via Homebrew (macOS):
```bash
brew install --cask miniforge
```

For **one-time use** in current terminal:
```bash
eval "$(mamba shell hook --shell zsh)"
```

For **permanent use** (adds to ~/.zshrc):
```bash
mamba init zsh
source ~/.zshrc
```

## Environment Management

Each paper implementation has its own isolated environment for perfect reproducibility.

### Quick Start
```bash
# Navigate to implementation
cd impls/slp

# Create environment
mamba env create -f environment.yml

# Activate environment
mamba activate slp

# Run implementation
python train.py
```

### Other Commands
```bash
# Run without activating (useful for scripts/automation)
mamba run -n slp python train.py

# Update environment
mamba env update -f environment.yml --prune

# Remove environment
mamba env remove -n slp

# List all environments
mamba env list
```

## Implementations

- **SLP** (`impls/slp/`) - Single Layer Perceptron implementation
