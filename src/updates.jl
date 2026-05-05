"""
Building-block update equations, matching the MATLAB hgf-toolbox exactly.
"""

# ──────────────────────────────────────────────────────────────────────
# Predictions
# ──────────────────────────────────────────────────────────────────────

"""Predicted mean at level j (with optional drift)."""
function hgf_prediction(mu_prev_j::Float64, t::Float64, rho_j::Float64)
    return mu_prev_j + t * rho_j
end

"""Predicted precision at intermediate level j."""
function hgf_pihat(pi_prev_j::Float64, t::Float64, ka_j::Float64,
                   mu_prev_jplus1::Float64, om_j::Float64)
    return 1.0 / (1.0 / pi_prev_j + t * exp(ka_j * mu_prev_jplus1 + om_j))
end

"""Predicted precision at the last level."""
function hgf_pihat_last(pi_prev_l::Float64, t::Float64, th::Float64)
    return 1.0 / (1.0 / pi_prev_l + t * th)
end

# ──────────────────────────────────────────────────────────────────────
# Level 1 (continuous input)
# ──────────────────────────────────────────────────────────────────────

"""
    hgf_continuous_level1(u, muhat1, pihat1, al)

Update level 1 given a continuous observation.  Returns (pi1, mu1, dau, da1).
"""
function hgf_continuous_level1(u::Float64, muhat1::Float64,
                               pihat1::Float64, al::Float64)
    dau = u - muhat1
    pi1 = pihat1 + 1.0 / al
    mu1 = muhat1 + (1.0 / pihat1) * (1.0 / (1.0 / pihat1 + al)) * dau
    da1 = (1.0 / pi1 + (mu1 - muhat1)^2) * pihat1 - 1.0
    return pi1, mu1, dau, da1
end

# ──────────────────────────────────────────────────────────────────────
# Volatility prediction error
# ──────────────────────────────────────────────────────────────────────

"""Volatility prediction error ``δ`` at any level."""
function hgf_volatility_pe(pi_j::Float64, mu_j::Float64,
                           muhat_j::Float64, pihat_j::Float64)
    return (1.0 / pi_j + (mu_j - muhat_j)^2) * pihat_j - 1.0
end

# ──────────────────────────────────────────────────────────────────────
# Volatility update — Classic HGF
# ──────────────────────────────────────────────────────────────────────

"""
    hgf_volatility_update(::ClassicUpdate, ...)

Standard HGF volatility update (Mathys et al., 2011).  Throws an error
if the posterior precision turns negative.

Returns `(pi_j, mu_j, v_jm1, w_jm1)`.
"""
function hgf_volatility_update(
        ::ClassicUpdate,
        muhat_j::Float64, pihat_j::Float64,
        ka_jm1::Float64, pihat_jm1::Float64, da_jm1::Float64,
        mu_prev_j::Float64, om_jm1::Float64, pi_prev_jm1::Float64,
        pi_jm1::Float64, mu_jm1::Float64, muhat_jm1::Float64,
        t::Float64)

    v_jm1 = t * exp(ka_jm1 * mu_prev_j + om_jm1)
    w_jm1 = v_jm1 * pihat_jm1

    pi_j = pihat_j + 0.5 * ka_jm1^2 * w_jm1 *
           (w_jm1 + (2.0 * w_jm1 - 1.0) * da_jm1)

    if pi_j <= 0.0
        return NaN, NaN, v_jm1, w_jm1   # signal failure
    end

    mu_j = muhat_j + 0.5 / pi_j * ka_jm1 * w_jm1 * da_jm1
    return pi_j, mu_j, v_jm1, w_jm1
end

# ──────────────────────────────────────────────────────────────────────
# Volatility update — Unbounded HGF
# ──────────────────────────────────────────────────────────────────────

"""
    hgf_volatility_update(::UHGFUpdate, ...)

Unbounded HGF volatility update — dual quadratic approximation
with Lambert W mode-finding and Gaussian mixture moment matching.

Expansion 1 is the quadratic expansion at the prediction (prior mean).
Expansion 2 is the quadratic expansion at the approximate posterior
mode obtained via the Lambert W₀ function, which solves the mode
equation exactly in the limit α → 0.

The two Gaussians are blended via a softmax weight based on the
variational energy I, and the final posterior is the moment-matched
Gaussian of the resulting two-component mixture.

Posterior precision is always positive.

Returns `(pi_j, mu_j, v_jm1, w_jm1)`.
"""
function hgf_volatility_update(
        ::UHGFUpdate,
        muhat_j::Float64, pihat_j::Float64,
        ka_jm1::Float64, pihat_jm1::Float64, da_jm1::Float64,
        mu_prev_j::Float64, om_jm1::Float64, pi_prev_jm1::Float64,
        pi_jm1::Float64, mu_jm1::Float64, muhat_jm1::Float64,
        t::Float64)

    # Recompute v and w using muhat_j (predicted mean) instead of mu_prev_j.
    # Rearranged to avoid Inf/Inf → NaN when v_jm1 overflows.
    v_jm1 = t * exp(ka_jm1 * muhat_j + om_jm1)
    w_jm1 = 1.0 / (1.0 + 1.0 / (pi_prev_jm1 * v_jm1))

    # ── Expansion 1: quadratic expansion at the prediction ──
    pi1 = pihat_j + 0.5 * ka_jm1^2 * w_jm1 * (1.0 - w_jm1)
    mu1 = muhat_j + 0.5 / pi1 * ka_jm1 * w_jm1 * da_jm1

    # ── Auxiliary quantities ──
    al_aux = 1.0 / pi_prev_jm1                              # σ†⁰
    be_aux = 1.0 / pi_jm1 + (mu_jm1 - muhat_jm1)^2         # total posterior uncertainty

    # ── Expansion 2: quadratic expansion at the Lambert W₀ approximate mode ──
    # In canonical variable y = log(t) + κx + ω, the variational energy is
    #   I(y) = −½ log(α + eʸ) − ½β/(α + eʸ) − ½π̂_y(y − γ_c)²
    # with γ_c = log(t) + κμ̂ + ω,  π̂_y = π̂/κ²,  β = be_aux,  α = σ†⁰.
    # Here σ̂ = 1/π̂ is a variance (not a standard deviation), as throughout
    # the HGF formulation, and π̂_y = π̂/κ² is the prior precision in y.
    # Setting α → 0 and solving I'(y) = 0 via the substitution
    #   v = (½ + π̂_y(y − γ_c)) / π̂_y
    # yields v·eᵛ = β/(2π̂_y)·exp(1/(2π̂_y) − γ_c), hence v = W₀(⋯) and
    #   y* = γ_c + v − 1/(2π̂_y).
    gamma_c = log(t) + ka_jm1 * muhat_j + om_jm1
    pihat_y = pihat_j / ka_jm1^2

    # Compute the Lambert W argument in log-space to avoid overflow in exp().
    log_W_arg = log(be_aux / (2.0 * pihat_y)) + 0.5 / pihat_y - gamma_c
    W_arg  = isfinite(log_W_arg) ? exp(min(log_W_arg, log(prevfloat(Inf)))) : NaN
    v_W    = _lambert_w0(W_arg)
    y_star = gamma_c + v_W - 0.5 / pihat_y
    x_star = (y_star - log(t) - om_jm1) / ka_jm1

    if isfinite(x_star)
        # Evaluate quadratic expansion at x_star. The w2/da2 forms are
        # rearranged to stay finite when s2 overflows.
        s2  = t * exp(ka_jm1 * x_star + om_jm1)
        w2  = 1.0 / (1.0 + al_aux / s2)
        da2 = be_aux / (al_aux + s2) - 1.0
        pi2 = pihat_j + 0.5 * ka_jm1^2 * w2 * (w2 + (2.0 * w2 - 1.0) * da2)
        if pi2 <= 0.0
            pi2 = pihat_j + 0.5 * ka_jm1^2 * w2 * (1.0 - w2)
        end
        mu2 = x_star + (0.5 * ka_jm1 * w2 * da2 - pihat_j * (x_star - muhat_j)) / pi2
    else
        # Expansion 2 is numerically unsafe; fall back to Expansion 1.
        pi2, mu2 = pi1, mu1
    end

    # ── Softmax blend weight based on variational energy I ──
    # Computed in log-space so the energies stay finite when eʸ overflows.
    log_al  = log(al_aux)
    log_be  = log(be_aux)
    log_t   = log(t)

    log_ey1  = log_t + ka_jm1 * mu1 + om_jm1
    log_sum1 = _logaddexp(log_al, log_ey1)          # log(α + eʸ¹)
    I1 = -0.5 * log_sum1 - 0.5 * exp(log_be - log_sum1) -
          0.5 * pihat_j * (mu1 - muhat_j)^2

    log_ey2  = log_t + ka_jm1 * mu2 + om_jm1
    log_sum2 = _logaddexp(log_al, log_ey2)
    I2 = -0.5 * log_sum2 - 0.5 * exp(log_be - log_sum2) -
          0.5 * pihat_j * (mu2 - muhat_j)^2

    # Stable sigmoid: 1/(1 + exp(d)) without overflowing exp for |d| ≫ 0.
    d = I1 - I2
    b = if isnan(d)
        0.5
    elseif d >= 0
        exp(-d) / (1.0 + exp(-d))
    else
        1.0 / (1.0 + exp(d))
    end

    # ── Gaussian mixture moment matching ──
    mu_j = (1.0 - b) * mu1 + b * mu2
    sig2 = (1.0 - b) / pi1 + b / pi2 + b * (1.0 - b) * (mu1 - mu2)^2
    pi_j = 1.0 / sig2

    return pi_j, mu_j, v_jm1, w_jm1
end

# ──────────────────────────────────────────────────────────────────────
# Numerical helpers
# ──────────────────────────────────────────────────────────────────────

"""Numerically stable ``\\log(e^a + e^b)``."""
_logaddexp(a::Float64, b::Float64) = a == b ? a + log(2.0) :
    max(a, b) + log1p(exp(-abs(a - b)))

# ──────────────────────────────────────────────────────────────────────
# Lambert W₀ function
# ──────────────────────────────────────────────────────────────────────

"""
    _lambert_w0(z)

Principal branch of the Lambert W function, W₀(z), for z ≥ 0.
Uses Halley's method which converges in 2–3 iterations to machine
precision for any positive argument.
"""
function _lambert_w0(z::Float64)
    z < 0.0 && return NaN
    z < 1e-10 && return z        # W(z) ≈ z for small z
    # Initial guess
    w = z > 3.0 ? log(z) - log(log(z)) : 1.0
    for _ in 1:8
        ew  = exp(w)
        f   = w * ew - z
        fp  = ew * (1.0 + w)
        fpp = ew * (2.0 + w)
        w  -= (2.0 * f * fp) / (2.0 * fp * fp - f * fpp)    # Halley
    end
    return w
end
