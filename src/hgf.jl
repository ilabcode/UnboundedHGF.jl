"""
    run_hgf(inputs, params, update_type) -> HGFTrajectory

Filter a vector of continuous observations `inputs` through an HGF with
the given `params` and `update_type` (ClassicUpdate() or UHGFUpdate()).

Returns an `HGFTrajectory`.  Most fields (`mu`, `pi`, `muhat`, `pihat`,
`v`, `da`) are T × L matrices; `w` is T × (L-1); `dau` is a length-T vector.

If ClassicUpdate encounters negative posterior precision it returns a
truncated trajectory up to the failure point, with `NaN` in the remaining rows.
"""
function run_hgf(inputs::AbstractVector{Float64}, params::HGFParams,
                 update_type::UpdateType)
    l = n_levels(params)
    @assert l >= 2 "Need at least 2 levels"
    @assert length(params.ka) == l - 1
    @assert length(params.om) == l - 1

    T = length(inputs)
    t_k = 1.0   # regular time intervals

    # Pre-allocate trajectory matrices (T × L)
    mu    = fill(NaN, T, l)
    pi_   = fill(NaN, T, l)
    muhat = fill(NaN, T, l)
    pihat = fill(NaN, T, l)
    v     = fill(NaN, T, l)
    w     = fill(NaN, T, l - 1)
    da    = fill(NaN, T, l)
    dau   = fill(NaN, T)

    # --- initial state ----
    pi_prev = 1.0 ./ params.sa_0       # precision = 1/variance
    mu_prev = copy(params.mu_0)

    for k in 1:T
        u_k = inputs[k]

        # ════════════════════════════════════════════════════════════
        # Level 1: continuous input update
        # ════════════════════════════════════════════════════════════
        muhat[k, 1] = hgf_prediction(mu_prev[1], t_k, params.rho[1])
        pihat[k, 1] = hgf_pihat(pi_prev[1], t_k, params.ka[1],
                                 mu_prev[2], params.om[1])

        pi_[k, 1], mu[k, 1], dau[k], da[k, 1] =
            hgf_continuous_level1(u_k, muhat[k, 1], pihat[k, 1], params.al)

        # ════════════════════════════════════════════════════════════
        # Intermediate levels 2 … l-1  (if l > 2)
        # ════════════════════════════════════════════════════════════
        for j in 2:(l - 1)
            muhat[k, j] = hgf_prediction(mu_prev[j], t_k, params.rho[j])
            pihat[k, j] = hgf_pihat(pi_prev[j], t_k, params.ka[j],
                                     mu_prev[j + 1], params.om[j])

            pi_[k, j], mu[k, j], v[k, j - 1], w[k, j - 1] =
                hgf_volatility_update(
                    update_type,
                    muhat[k, j], pihat[k, j],
                    params.ka[j - 1], pihat[k, j - 1], da[k, j - 1],
                    mu_prev[j], params.om[j - 1], pi_prev[j - 1],
                    pi_[k, j - 1], mu[k, j - 1], muhat[k, j - 1],
                    t_k)

            if isnan(pi_[k, j])
                return HGFTrajectory(mu, pi_, muhat, pihat, v, w, da, dau)
            end

            da[k, j] = hgf_volatility_pe(pi_[k, j], mu[k, j],
                                          muhat[k, j], pihat[k, j])
        end

        # ════════════════════════════════════════════════════════════
        # Last level l
        # ════════════════════════════════════════════════════════════
        muhat[k, l] = hgf_prediction(mu_prev[l], t_k, params.rho[l])
        pihat[k, l] = hgf_pihat_last(pi_prev[l], t_k, params.th)

        # v at level l (for storage)
        v[k, l] = t_k * params.th
        if update_type isa UHGFUpdate
            v[k, l - 1] = t_k * exp(params.ka[l - 1] * muhat[k, l] +
                                     params.om[l - 1])
        else
            v[k, l - 1] = t_k * exp(params.ka[l - 1] * mu_prev[l] +
                                     params.om[l - 1])
        end

        pi_[k, l], mu[k, l], _, w[k, l - 1] =
            hgf_volatility_update(
                update_type,
                muhat[k, l], pihat[k, l],
                params.ka[l - 1], pihat[k, l - 1], da[k, l - 1],
                mu_prev[l], params.om[l - 1], pi_prev[l - 1],
                pi_[k, l - 1], mu[k, l - 1], muhat[k, l - 1],
                t_k)

        if isnan(pi_[k, l])
            return HGFTrajectory(mu, pi_, muhat, pihat, v, w, da, dau)
        end

        da[k, l] = hgf_volatility_pe(pi_[k, l], mu[k, l],
                                      muhat[k, l], pihat[k, l])

        # ── advance state ──
        mu_prev .= mu[k, :]
        pi_prev .= pi_[k, :]
    end

    return HGFTrajectory(mu, pi_, muhat, pihat, v, w, da, dau)
end

"""
    first_nan_row(traj::HGFTrajectory) -> Int

Return the first row where any posterior mean is NaN (i.e., where the
classic HGF crashed), or 0 if none.
"""
function first_nan_row(traj::HGFTrajectory)
    for k in 1:size(traj.mu, 1)
        if any(isnan, @view traj.mu[k, :])
            return k
        end
    end
    return 0
end
