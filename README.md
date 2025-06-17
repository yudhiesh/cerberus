# Cerberus: Safeguards for LLMs

This repository contains the implementation of the synthetic data generation pipeline described in [Cerberus: Safeguards for Large Language Models - Synthetic Data Generation (Part 1)](https://yudhiesh.github.io/2025/06/16/cerberus-safeguards-for-llms-synthetic-data-generation-part-1.html). 

## Development Setup

This project uses Nix to manage development dependencies. Follow these steps to get started:

## Prerequisites

### Install Nix

**On Linux/macOS:**
```bash
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```

**Alternative (official installer):**
```bash
sh <(curl -L https://nixos.org/nix/install) --daemon
```

## Setup Steps

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-project>
   ```

2. **Enter the development environment:**
   ```bash
   nix-shell
   ```
   
   This will automatically download and install all required tools (just, act, uv).

3. **Verify installation:**
   The shell will show version information for all tools when you enter the environment.

## Usage

Once in the development environment, you have access to:

- **`just`** - Command runner for project tasks
- **`act`** - Run GitHub Actions locally  
- **`uv`** - Fast Python package management
