"""
Update type selectors for the HGF volatility precision update.
"""
abstract type UpdateType end

"""Classic HGF (Mathys et al., 2011) — can produce negative posterior precision."""
struct ClassicUpdate <: UpdateType end

"""Unbounded HGF — dual quadratic approximation; never produces negative precision."""
struct UHGFUpdate <: UpdateType end

"""
    HGFParams

Parameters for an n-level continuous HGF.
"""
struct HGFParams
    mu_0::Vector{Float64}   # initial means  (length n_levels)
    sa_0::Vector{Float64}   # initial variances (length n_levels)
    rho::Vector{Float64}    # drift rates  (length n_levels)
    ka::Vector{Float64}     # coupling strengths κ  (length n_levels-1)
    om::Vector{Float64}     # log-volatility offsets ω  (length n_levels-1)
    th::Float64             # meta-volatility θ at the last level
    al::Float64             # observation noise variance α = 1/π_u
end

function n_levels(p::HGFParams)
    return length(p.mu_0)
end

"""
    HGFTrajectory

Stores the full trajectory of beliefs across time.  `T` = number of
observations, `L` = number of levels.

- `mu`, `pi`, `muhat`, `pihat`, `v`, `da` — T × L matrices
- `w` — T × (L-1) matrix (one coupling per adjacent level pair)
- `dau` — length-T vector (unsigned prediction error at level 1)
"""
struct HGFTrajectory
    mu::Matrix{Float64}
    pi::Matrix{Float64}
    muhat::Matrix{Float64}
    pihat::Matrix{Float64}
    v::Matrix{Float64}
    w::Matrix{Float64}
    da::Matrix{Float64}
    dau::Vector{Float64}
end
