# UnboundedHGF.jl

A lean Julia reference implementation of robust volatility updates for Hierarchical Gaussian Filtering (HGF). The new update equations never produce negative posterior precision and faithfully track the variational posterior even for large prediction errors.

This package accompanies the technical report:

> Mathys, C., Legrand, N., Waade, P. T., Mikus, N., & Weber, L. A. (2026). *Robust volatility updates for Hierarchical Gaussian Filtering.* arXiv:2605.00966 [cs.LG]. https://doi.org/10.48550/arXiv.2605.00966

---

## Background

HGF networks model an agent's inference on hidden environmental states through a cascade of precision-weighted prediction errors. Volatility-coupled nodes update their precision (inverse variance) via a quadratic approximation to the variational energy. In some regions of parameter space the original approximation produces negative posterior precision — a logical impossibility that terminates filtering with an error.

This package introduces a **dual quadratic approximation** that solves the problem:

1. Two quadratic expansions of the variational energy are computed: one at the prior prediction and one at a second mode whose location is obtained in closed form via the **Lambert W function**.
2. The two expansions are blended with a softmax weight that favours whichever approximation is more faithful.
3. Gaussian mixture moment matching yields a final posterior with guaranteed positive precision.

The resulting `UHGFUpdate` equations are robust across the entire parameter space.

---

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/ilabcode/UnboundedHGF.jl")
```

Requires Julia ≥ 1.11.

---

## Quick start

```julia
using UnboundedHGF

# Two-level HGF parameters
params = HGFParams(
    mu_0 = [0.0, 0.0],     # initial means
    sa_0 = [1.0, 1.0],     # initial variances
    rho  = [0.0, 0.0],     # drift rates
    ka   = [1.0],           # coupling strength κ
    om   = [-4.0],          # log-volatility offset ω
    th   = 1e-4,            # meta-volatility θ
    al   = 1e-4,            # observation noise variance α
)

inputs = randn(200)  # continuous observations

# Robust update (never crashes)
traj_u = run_hgf(inputs, params, UHGFUpdate())

# Classic update (may return truncated trajectory on negative precision)
traj_c = run_hgf(inputs, params, ClassicUpdate())

# Posterior means and precisions (T × L matrices)
traj_u.mu    # posterior means
traj_u.pi    # posterior precisions
```

---

## API

### Types

| Type | Description |
|------|-------------|
| `HGFParams` | Parameters for an n-level continuous HGF |
| `HGFTrajectory` | Full belief trajectory returned by `run_hgf` |
| `UHGFUpdate` | Robust dual-approximation volatility update |
| `ClassicUpdate` | Original HGF volatility update (Mathys et al., 2011) |

#### `HGFParams` fields

| Field | Type | Description |
|-------|------|-------------|
| `mu_0` | `Vector{Float64}` | Initial means (length `n_levels`) |
| `sa_0` | `Vector{Float64}` | Initial variances (length `n_levels`) |
| `rho`  | `Vector{Float64}` | Drift rates (length `n_levels`) |
| `ka`   | `Vector{Float64}` | Coupling strengths κ (length `n_levels - 1`) |
| `om`   | `Vector{Float64}` | Log-volatility offsets ω (length `n_levels - 1`) |
| `th`   | `Float64` | Meta-volatility θ at the last level |
| `al`   | `Float64` | Observation noise variance α = 1/π_u |

#### `HGFTrajectory` fields

| Field | Dimensions | Description |
|-------|-----------|-------------|
| `mu`     | T × L     | Posterior means |
| `pi`     | T × L     | Posterior precisions |
| `muhat`  | T × L     | Predicted means |
| `pihat`  | T × L     | Predicted precisions |
| `v`      | T × L     | Predicted variances |
| `da`     | T × L     | Prediction errors |
| `w`      | T × (L-1) | Volatility weights |
| `dau`    | length T  | Unsigned prediction error at level 1 |

### Functions

| Function | Description |
|----------|-------------|
| `run_hgf(inputs, params, update_type)` | Filter observations; returns `HGFTrajectory` |
| `n_levels(params)` | Number of levels in the HGF |
| `first_nan_row(traj)` | Index of the first row containing `NaN` (0 if none) |

---

## Repository structure

```
src/
  UnboundedHGF.jl   module entry point
  types.jl          HGFParams, HGFTrajectory, update type selectors
  updates.jl        all building-block update equations
  hgf.jl            run_hgf filtering loop
notebooks/
  unbounded.jl      Pluto notebook — full theory derivation and simulations
scripts/
  generate_figures.jl  reproduces all figures from the accompanying paper
data/
  suff_stat.csv     time series used in simulations
```

---

## Reproducing the paper figures

With Julia and the package dependencies installed, run:

```bash
julia --project scripts/generate_figures.jl
```

Figures are written to the path configured in the script as both PDF and PNG.
