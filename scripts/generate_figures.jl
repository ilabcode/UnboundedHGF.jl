#!/usr/bin/env julia
#
# Generate ALL figures for the uHGF paper:
#
#   fig:origapprox  — four panels: Jderivs and Japprox for two parameter sets
#   fig:Jcomp       — components of the variational energy
#   fig:bimodal     — bimodal variational posterior with two-expansion overlay
#   sim1            — KL-divergence heat map (classic vs uHGF)
#   sim2            — Agreement on suff_stat data (moderate volatility)
#   sim3            — Classic crashes on suff_stat data
#   sim4            — Parameter-space coverage map
#
# Run from the UnboundedHGF.jl root:
#   julia --project=. scripts/generate_figures.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using UnboundedHGF
using Plots
using LaTeXStrings
using Printf
using DelimitedFiles
using Statistics: mean

gr()

const FIGDIR  = joinpath(@__DIR__, "..", "..", "uhgf-paper", "figures")
const DATADIR = joinpath(@__DIR__, "..", "data")

mkpath(FIGDIR)

# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════

function load_suff_stat()
    path = joinpath(DATADIR, "suff_stat.csv")
    return readdlm(path, ',', Float64)
end

# ═══════════════════════════════════════════════════════════════════════
# Math functions (canonical variational energy and approximations)
# ═══════════════════════════════════════════════════════════════════════

# ── Variational energy and derivatives (exact, from the Pluto notebook) ──

w(α, x)       = exp(x) / (α + exp(x))
δ(α, β, x)   = β / (α + exp(x)) - 1.0
J(x, α, β, γ) = -0.5 * log(α + exp(x)) - 0.5 * β / (α + exp(x)) - 0.25 * (x - γ)^2
dJ(x, α, β, γ) = 0.5 * w(α, x) * δ(α, β, x) - 0.5 * (x - γ)
ddJ(x, α, β)   = -0.5 * w(α, x) * (w(α, x) + (2 * w(α, x) - 1) * δ(α, β, x)) - 0.5
Jhat(x, α, β, γ) = J(γ, α, β, γ) + dJ(γ, α, β, γ) * (x - γ) + 0.5 * ddJ(γ, α, β) * (x - γ)^2

# ── Component decomposition (for fig:Jcomp) ──

J1(x, α)    = -0.5 * log(α + exp(x))
J2(x, α, β) = -0.5 * β / (α + exp(x))
J3(x, γ)    = -0.25 * (x - γ)^2

# ── Canonical variational energy (alias for simulation code) ──

J_canon(x, α, β, γ) = J(x, α, β, γ)
p_unnorm(x, α, β, γ) = exp(J_canon(x, α, β, γ))

function trap_integrate(f, a, b; n=20_000)
    xs = range(a, b; length=n)
    dx = xs[2] - xs[1]
    vals = f.(xs)
    return dx * (0.5 * vals[1] + sum(vals[2:end-1]) + 0.5 * vals[end])
end

function p_var_posterior(xs, α, β, γ; lo=-80.0, hi=80.0)
    Z = trap_integrate(x -> p_unnorm(x, α, β, γ), lo, hi)
    return [p_unnorm(x, α, β, γ) / Z for x in xs]
end

w_canon(α, x)    = w(α, x)
δ_canon(α, β, x) = δ(α, β, x)

function classic_approx(α, β, γ)
    wg = w_canon(α, γ)
    dg = δ_canon(α, β, γ)
    π_J = 0.5 * wg * (wg + (2wg - 1) * dg) + 0.5
    if π_J <= 0
        return NaN, NaN
    end
    μ_J = γ + 0.5 / π_J * wg * dg
    return μ_J, π_J
end

"""Lambert W₀ via Halley's method (converges in 2–3 iterations for z > 0)."""
function _lambert_w0(z::Float64)
    z < 0.0  && return NaN
    z < 1e-10 && return z
    w = z > 3.0 ? log(z) - log(log(z)) : 1.0
    for _ in 1:8
        ew  = exp(w)
        f   = w * ew - z
        fp  = ew * (1.0 + w)
        fpp = ew * (2.0 + w)
        w  -= (2.0 * f * fp) / (2.0 * fp * fp - f * fpp)
    end
    return w
end

function uhgf_approx(α, β, γ)
    wg = w_canon(α, γ)
    dg = δ_canon(α, β, γ)

    π_L1 = 0.5 * wg * (1.0 - wg) + 0.5
    μ_L1 = γ + 0.5 / π_L1 * wg * dg

    W_arg  = β * exp(1.0 - γ)
    y_star = γ - 1.0 + _lambert_w0(W_arg)

    s2   = exp(y_star)
    w2   = s2 / (α + s2)
    d2   = β / (α + s2) - 1.0
    π_L2 = 0.5 * w2 * (w2 + (2w2 - 1) * d2) + 0.5
    if π_L2 <= 0.0
        π_L2 = 0.5 * w2 * (1.0 - w2) + 0.5
    end
    μ_L2 = y_star + (0.5 * w2 * d2 - 0.5 * (y_star - γ)) / π_L2

    J1_ = J_canon(μ_L1, α, β, γ)
    J2_ = J_canon(μ_L2, α, β, γ)
    b   = 1.0 / (1.0 + exp(J1_ - J2_))

    μ  = (1.0 - b) * μ_L1 + b * μ_L2
    σ² = (1.0 - b) / π_L1 + b / π_L2 + b * (1.0 - b) * (μ_L1 - μ_L2)^2
    return μ, 1.0 / σ²
end

function kl_divergence_numerical(xs, p_vals, μ_q, π_q)
    dx        = xs[2] - xs[1]
    kl        = 0.0
    log_norm_q = 0.5 * log(π_q / (2π))
    for (i, x) in enumerate(xs)
        p = p_vals[i]
        if p > 1e-300
            log_q = log_norm_q - 0.5 * π_q * (x - μ_q)^2
            kl += p * (log(p) - log_q) * dx
        end
    end
    return max(kl, 0.0)
end

# ═══════════════════════════════════════════════════════════════════════
# Figure: origapprox (four panels)  — fig:origapprox
#
# Exactly reproduces the Pluto notebook plots that were previously
# exported to PNG. Style: plain Unicode titles, default GR palette,
# xlabel="x", ylabel="Value", xlimits=(-25,25), ylimits=(-200,100).
# ═══════════════════════════════════════════════════════════════════════

function figure_origapprox()
    println("Generating fig:origapprox …")

    x = range(-60, 60; step=0.02)

    for (α, β, γ, tag) in [
            (1.0,   1.0, -6.0, "a1-b1-gm6"),
            (0.005, 1.0, -6.0, "a0_0005-b1-gm6"),
        ]

        title_str = "(α, β, γ) = ($α, $β, $(Int(γ)))"

        # ── Derivatives panel ──
        fig_d = plot(x, [J.(x, α, β, γ)   dJ.(x, α, β, γ)   ddJ.(x, α, β)];
            title    = title_str,
            xlabel   = "x",
            ylabel   = "Value",
            labels   = ["J(x)" "J'(x)" "J''(x)"],
            xlimits  = (-25, 25),
            ylimits  = (-200, 100))

        savefig(fig_d, joinpath(FIGDIR, "Jderivs-$tag.pdf"))
        savefig(fig_d, joinpath(FIGDIR, "Jderivs-$tag.png"))

        # ── Approximation panel ──
        fig_a = plot(x, [J.(x, α, β, γ)   Jhat.(x, α, β, γ)];
            title    = title_str,
            xlabel   = "x",
            ylabel   = "Value",
            labels   = ["J(x)" "Jhat(x)"],
            xlimits  = (-25, 25),
            ylimits  = (-200, 100))
        plot!(fig_a, [γ]; seriestype=:vline, labels="γ")

        savefig(fig_a, joinpath(FIGDIR, "Japprox-$tag.pdf"))
        savefig(fig_a, joinpath(FIGDIR, "Japprox-$tag.png"))
    end

    println("  Saved Jderivs-*.pdf/png and Japprox-*.pdf/png")
end

# ═══════════════════════════════════════════════════════════════════════
# Figure: J components  — fig:Jcomp
# ═══════════════════════════════════════════════════════════════════════

function figure_Jcomponents()
    println("Generating fig:Jcomp …")

    α = 0.005
    β = 1.0
    γ = -6.0
    xs = range(-25, 25; step=0.02)

    fig = plot(xs, [J.(xs, α, β, γ)  J1.(xs, α)  J2.(xs, α, β)  J3.(xs, γ)];
        title    = "(α, β, γ) = (0.005, 1, -6)",
        xlabel   = "x",
        ylabel   = "Value",
        labels   = ["J(x)" "J1(x)" "J2(x)" "J3(x)"],
        xlimits  = (-25, 25),
        ylimits  = (-200, 100))

    savefig(fig, joinpath(FIGDIR, "Jcomponents.pdf"))
    savefig(fig, joinpath(FIGDIR, "Jcomponents.png"))
    println("  Saved Jcomponents.pdf/png")
end

# ═══════════════════════════════════════════════════════════════════════
# Figure: Bimodal variational posterior  — fig:bimodal
# ═══════════════════════════════════════════════════════════════════════

function figure_bimodal()
    println("Generating fig:bimodal …")

    α = 0.05
    β = 1.0
    γ = -7.0

    xs    = range(-25, 25; step=0.02)
    jvals = J.(xs, α, β, γ)

    mode_indices = Int[]
    for i in 2:length(xs)-1
        if jvals[i] > jvals[i-1] && jvals[i] > jvals[i+1]
            push!(mode_indices, i)
        end
    end

    jmax    = maximum(jvals)
    q_unnorm = exp.(jvals .- jmax)
    dx      = xs[2] - xs[1]
    Z       = sum(q_unnorm) * dx
    q_vals  = q_unnorm ./ Z

    p1 = plot(xs, jvals;
        title    = "(α, β, γ) = (0.05, 1, -7)",
        xlabel   = "x",
        ylabel   = "Value",
        labels   = "J(x)",
        xlimits  = (-15, 10),
        ylimits  = (-20, 0))
    for (k, idx) in enumerate(mode_indices)
        scatter!(p1, [xs[idx]], [jvals[idx]];
            label      = (k == 1 ? "Local maxima" : nothing),
            color      = :red,
            markersize = 5)
    end

    p2 = plot(xs, q_vals;
        title    = "Variational posterior q(x)",
        xlabel   = "x",
        ylabel   = "q(x)",
        labels   = "q(x)",
        xlimits  = (-15, 10))
    for (k, idx) in enumerate(mode_indices)
        scatter!(p2, [xs[idx]], [q_vals[idx]];
            label      = (k == 1 ? "Modes" : nothing),
            color      = :red,
            markersize = 5)
    end

    fig = plot(p1, p2; layout=(2, 1), size=(600, 500))

    savefig(fig, joinpath(FIGDIR, "Jbimodal.pdf"))
    savefig(fig, joinpath(FIGDIR, "Jbimodal.png"))
    println("  Saved Jbimodal.pdf/png")
end

# ═══════════════════════════════════════════════════════════════════════
# Simulation 1: KL-divergence heat map
# ═══════════════════════════════════════════════════════════════════════

function run_simulation_1()
    println("=" ^ 72)
    println("SIMULATION 1: KL divergence heat map")
    println("=" ^ 72)

    α_val  = 0.005
    ratios = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    γ_vals = collect(range(-15.0, 15.0; length=61))
    xs     = collect(range(-60.0, 60.0; length=4001))

    nr = length(ratios)
    ng = length(γ_vals)

    kl_classic = fill(NaN, nr, ng)
    kl_uhgf    = fill(NaN, nr, ng)

    for (i, ratio) in enumerate(ratios)
        β_val = ratio * α_val
        for (j, γ) in enumerate(γ_vals)
            p_vals = p_var_posterior(xs, α_val, β_val, γ)

            μ_c, π_c = classic_approx(α_val, β_val, γ)
            μ_u, π_u = uhgf_approx(α_val, β_val, γ)

            kl_uhgf[i, j] = kl_divergence_numerical(xs, p_vals, μ_u, π_u)

            if !isnan(π_c) && π_c > 0
                kl_classic[i, j] = kl_divergence_numerical(xs, p_vals, μ_c, π_c)
            end
        end
    end

    valid_classic  = .!isnan.(kl_classic)
    n_total        = nr * ng
    n_classic_ok   = sum(valid_classic)
    n_classic_fail = n_total - n_classic_ok
    @printf("  Total: %d,  Classic OK: %d (%.1f%%),  Classic fail: %d (%.1f%%)\n",
            n_total, n_classic_ok, 100n_classic_ok / n_total,
            n_classic_fail, 100n_classic_fail / n_total)
    @printf("  Mean KL classic (where ok): %.4f\n",
            mean(kl_classic[valid_classic]))
    @printf("  Mean KL uhgf:               %.4f\n", mean(kl_uhgf))

    # Clamp to [0, 5]; mark classic failures as 5 (max heat)
    kl_c_plot = clamp.(kl_classic, 0.0, 5.0)
    kl_c_plot[isnan.(kl_c_plot)] .= 5.0
    kl_u_plot = clamp.(kl_uhgf, 0.0, 5.0)

    ratio_labels = string.(ratios)

    p1 = heatmap(γ_vals, 1:nr, kl_c_plot;
                 xlabel          = L"\gamma",
                 ylabel          = L"\beta/\alpha",
                 yticks          = (1:nr, ratio_labels),
                 title           = "Classic HGF",
                 titlefontsize   = 9,
                 colorbar_title  = L"D_\mathrm{KL}",
                 clims           = (0, 5),
                 color           = :viridis,
                 size            = (420, 300))

    p2 = heatmap(γ_vals, 1:nr, kl_u_plot;
                 xlabel          = L"\gamma",
                 ylabel          = L"\beta/\alpha",
                 yticks          = (1:nr, ratio_labels),
                 title           = "uHGF",
                 titlefontsize   = 9,
                 colorbar_title  = L"D_\mathrm{KL}",
                 clims           = (0, 5),
                 color           = :viridis,
                 size            = (420, 300))

    fig = plot(p1, p2;
               layout               = (1, 2),
               size                 = (900, 320),
               plot_title           = "Approximation quality",
               plot_titlefontsize   = 10,
               plot_titlevspan      = 0.06,
               plot_titlefontweight = :bold,
               margin               = 5Plots.mm,
               bottom_margin        = 7Plots.mm)

    savefig(fig, joinpath(FIGDIR, "sim1-kl-heatmap.pdf"))
    println("  Saved sim1-kl-heatmap.pdf\n")
end

# ═══════════════════════════════════════════════════════════════════════
# Simulation 2: Agreement on suff_stat data (moderate volatility)
# ═══════════════════════════════════════════════════════════════════════

function run_simulation_2()
    println("=" ^ 72)
    println("SIMULATION 2: Agreement on suff_stat data")
    println("=" ^ 72)

    data    = load_suff_stat()
    u       = data[:, 1]
    mu_true = data[:, 2]
    sd_true = data[:, 3]
    T       = length(u)

    params = HGFParams(
        [0.0, 1.0],
        [200.0, 1.0],
        [0.0, 0.0],
        [1.0],
        [2.0],
        exp(-1.0),
        1000.0,
    )

    traj_c = run_hgf(u, params, ClassicUpdate())
    traj_u = run_hgf(u, params, UHGFUpdate())

    crash_c = first_nan_row(traj_c)
    crash_u = first_nan_row(traj_u)
    @printf("  Classic: %s    uHGF: %s\n",
            crash_c == 0 ? "OK" : "CRASHED at $(crash_c)",
            crash_u == 0 ? "OK" : "CRASHED at $(crash_u)")

    if crash_c == 0 && crash_u == 0
        for j in 1:2
            d_mu = maximum(abs.(traj_c.mu[:, j] .- traj_u.mu[:, j]))
            rmse = sqrt(mean((traj_c.mu[:, j] .- traj_u.mu[:, j]).^2))
            @printf("  Level %d:  max|Δμ|=%.2e  RMSE(μ)=%.2e\n", j, d_mu, rmse)
        end
    end

    ks   = 1:T
    sa_c = 1.0 ./ traj_c.pi
    sa_u = 1.0 ./ traj_u.pi

    p1 = plot(; xlabel="Step", ylabel=L"\mu_1",
              xguidefontsize=8,
              title="Level 1", titlefontsize=9, legend=:topleft, legendfontsize=6)
    plot!(p1, ks, mu_true; ribbon=2 .* sd_true,
          fillalpha=0.15, fillcolor=:grey, linecolor=:grey,
          linewidth=0.5, label="Ground truth ± 2 SD")
    scatter!(p1, ks, u; markersize=1.5, markercolor=:black,
             markerstrokewidth=0, alpha=0.5, label="Observations")
    plot!(p1, ks, traj_c.mu[:, 1]; ribbon=2 .* sqrt.(sa_c[:, 1]),
          fillalpha=0.15, fillcolor=:blue, linewidth=1.5, color=:blue,
          label="Classic HGF")
    plot!(p1, ks, traj_u.mu[:, 1]; ribbon=2 .* sqrt.(sa_u[:, 1]),
          fillalpha=0.15, fillcolor=:red, linewidth=1.5, color=:red,
          label="uHGF")

    p2 = plot(; xlabel="Step", ylabel=L"\mu_2",
              xguidefontsize=8,
              title="Level 2", titlefontsize=9, legend=:topleft, legendfontsize=6)
    plot!(p2, ks, traj_c.mu[:, 2]; ribbon=2 .* sqrt.(sa_c[:, 2]),
          fillalpha=0.15, fillcolor=:blue, linewidth=1.5, color=:blue,
          label="Classic HGF")
    plot!(p2, ks, traj_u.mu[:, 2]; ribbon=2 .* sqrt.(sa_u[:, 2]),
          fillalpha=0.15, fillcolor=:red, linewidth=1.5, color=:red,
          label="uHGF")

    param_str = @sprintf("\$\\omega_1\$ = %.1f,  \$\\omega_2\$ = %.1f,  \$\\alpha_u\$ = %.0f",
                         params.om[1], log(params.th), params.al)
    fig = plot(p1, p2; layout=(2, 1), size=(700, 500),
               plot_title    = "Filtering under moderate volatility\n\n" * param_str,
               plot_titlefontsize   = 9, plot_titlevspan = 0.10,
               plot_titlefontweight = :bold,
               margin=4Plots.mm, left_margin=6Plots.mm)

    savefig(fig, joinpath(FIGDIR, "sim2-agreement.pdf"))
    println("  Saved sim2-agreement.pdf\n")
end

# ═══════════════════════════════════════════════════════════════════════
# Simulation 3: Classic crashes on suff_stat data
# ═══════════════════════════════════════════════════════════════════════

function run_simulation_3()
    println("=" ^ 72)
    println("SIMULATION 3: Classic crashes on suff_stat data")
    println("=" ^ 72)

    data    = load_suff_stat()
    u       = data[:, 1]
    mu_true = data[:, 2]
    sd_true = data[:, 3]
    T       = length(u)

    params = HGFParams(
        [0.0, 1.0],
        [200.0, 1.0],
        [0.0, 0.0],
        [1.0],
        [2.0],
        exp(2.0),
        1000.0,
    )
    @printf("\n  Using: om=%.2f, sa2_0=%.4f, th=%.6f, al=%.1f\n",
            params.om[1], params.sa_0[2], params.th, params.al)

    traj_c  = run_hgf(u, params, ClassicUpdate())
    traj_u  = run_hgf(u, params, UHGFUpdate())
    crash_c = first_nan_row(traj_c)
    crash_u = first_nan_row(traj_u)

    @printf("  Classic: %s\n", crash_c > 0 ? "CRASHED at step $(crash_c)" : "OK")
    @printf("  uHGF:    %s\n", crash_u == 0 ? "OK — all $(T) steps" : "CRASHED at $(crash_u)")

    if crash_u == 0
        min_pi2 = minimum(traj_u.pi[:, 2])
        @printf("  Min pi_2 (uHGF): %.6f\n", min_pi2)
    end

    ks      = 1:T
    valid_c = crash_c > 0 ? crash_c - 1 : T

    p1 = plot(; xlabel="Step", ylabel=L"\mu_1",
              xguidefontsize=8,
              title="Level 1", titlefontsize=9, legend=:topleft, legendfontsize=6)
    plot!(p1, ks, mu_true; ribbon=2 .* sd_true,
          fillalpha=0.15, fillcolor=:grey, linecolor=:grey,
          linewidth=0.5, label="Ground truth ± 2 SD")
    scatter!(p1, ks, u; markersize=1.5, markercolor=:black,
             markerstrokewidth=0, alpha=0.5, label="Observations")
    if valid_c > 0
        sa_c_valid = 1.0 ./ traj_c.pi[1:valid_c, :]
        plot!(p1, 1:valid_c, traj_c.mu[1:valid_c, 1];
              ribbon=2 .* sqrt.(sa_c_valid[:, 1]),
              fillalpha=0.15, fillcolor=:blue, linewidth=1.5,
              color=:blue, label="Classic HGF")
    end
    if crash_c > 0
        vline!(p1, [crash_c]; color=:blue, linestyle=:dot,
               linewidth=1.0, label="Classic crash")
    end
    sa_u = 1.0 ./ traj_u.pi
    plot!(p1, ks, traj_u.mu[:, 1]; ribbon=2 .* sqrt.(sa_u[:, 1]),
          fillalpha=0.15, fillcolor=:red, linewidth=1.5, color=:red,
          label="uHGF")

    p2 = plot(; xlabel="Step", ylabel=L"\mu_2",
              xguidefontsize=8,
              title="Level 2", titlefontsize=9, legend=:topleft, legendfontsize=6)
    if valid_c > 0
        plot!(p2, 1:valid_c, traj_c.mu[1:valid_c, 2];
              ribbon=2 .* sqrt.(sa_c_valid[:, 2]),
              fillalpha=0.15, fillcolor=:blue, linewidth=1.5,
              color=:blue, label="Classic HGF")
    end
    if crash_c > 0
        vline!(p2, [crash_c]; color=:blue, linestyle=:dot,
               linewidth=1.0, label="Classic crash")
    end
    plot!(p2, ks, traj_u.mu[:, 2]; ribbon=2 .* sqrt.(sa_u[:, 2]),
          fillalpha=0.15, fillcolor=:red, linewidth=1.5, color=:red,
          label="uHGF")

    param_str = @sprintf("\$\\omega_1\$ = %.1f,  \$\\omega_2\$ = %.1f,  \$\\alpha_u\$ = %.0f",
                         params.om[1], log(params.th), params.al)
    fig = plot(p1, p2; layout=(2, 1), size=(700, 500),
               plot_title    = "Robustness under extreme prediction errors\n\n" * param_str,
               plot_titlefontsize   = 9, plot_titlevspan = 0.10,
               plot_titlefontweight = :bold,
               margin=4Plots.mm, left_margin=6Plots.mm)

    savefig(fig, joinpath(FIGDIR, "sim3-crash.pdf"))
    println("  Saved sim3-crash.pdf\n")

    @printf("  Paper parameters: om=%.2f, sa2_0=%.4f, th=%.6f (=exp(%.2f)), al=%.1f\n",
            params.om[1], params.sa_0[2], params.th, log(params.th), params.al)
end

# ═══════════════════════════════════════════════════════════════════════
# Simulation 4: Parameter-space coverage map
# ═══════════════════════════════════════════════════════════════════════

function run_simulation_4()
    println("=" ^ 72)
    println("SIMULATION 4: Parameter-space coverage map")
    println("=" ^ 72)

    data = load_suff_stat()
    u    = data[:, 1]

    om_grid  = collect(range(-16.0, 2.0; step=0.1))
    lth_grid = collect(range(-16.0, 2.0; step=0.1))

    n_om  = length(om_grid)
    n_lth = length(lth_grid)
    n_total = n_om * n_lth

    # 0 = both ok, 1 = classic fails (uhgf ok), 2 = both fail, 3 = unexpected
    status = zeros(Int, n_om, n_lth)

    n_classic_ok = 0
    n_uhgf_ok    = 0
    cnt          = 0

    @printf("  Scanning %d × %d = %d parameter combinations\n", n_om, n_lth, n_total)

    for (i, om) in enumerate(om_grid)
        for (j, lth) in enumerate(lth_grid)
            cnt += 1
            if cnt % 500 == 0
                @printf("  %d / %d (%.0f%%)\r", cnt, n_total, 100cnt / n_total)
            end

            params = HGFParams(
                [u[1], 0.0],
                [100.0, 1.0],
                [0.0, 0.0],
                [1.0],
                [om],
                exp(lth),
                100.0,
            )

            traj_c    = run_hgf(u, params, ClassicUpdate())
            classic_ok = first_nan_row(traj_c) == 0

            traj_u   = run_hgf(u, params, UHGFUpdate())
            uhgf_ok  = first_nan_row(traj_u) == 0

            if classic_ok;  n_classic_ok += 1; end
            if uhgf_ok;     n_uhgf_ok    += 1; end

            if classic_ok && uhgf_ok
                status[i, j] = 0
            elseif !classic_ok && uhgf_ok
                status[i, j] = 1
            elseif !classic_ok && !uhgf_ok
                status[i, j] = 2
            else
                status[i, j] = 3
            end
        end
    end

    @printf("\n  Total: %d\n", n_total)
    @printf("  Classic OK: %d (%.1f%%)\n", n_classic_ok, 100n_classic_ok / n_total)
    @printf("  uHGF OK:    %d (%.1f%%)\n", n_uhgf_ok, 100n_uhgf_ok / n_total)
    @printf("  Classic-only failures: %d (%.1f%%)\n",
            sum(status .== 1), 100sum(status .== 1) / n_total)

    plot_mat = Float64.(status')

    fig = heatmap(om_grid, lth_grid, plot_mat;
                  xlabel    = L"\omega_1",
                  ylabel    = L"\omega_2",
                  color     = cgrad([:steelblue, :orange], [0.5]),
                  clims     = (-0.5, 1.5),
                  colorbar  = false,
                  size      = (500, 420),
                  title     = "Parameter-space coverage",
                  titlefontsize       = 9,
                  titlefontweight     = :bold)

    scatter!(fig, [NaN], [NaN]; color=:steelblue, markersize=6,
             markerstrokewidth=0, label="Both succeed")
    scatter!(fig, [NaN], [NaN]; color=:orange, markersize=6,
             markerstrokewidth=0, label="Classic HGF fails")
    plot!(fig; legend=:topright)

    savefig(fig, joinpath(FIGDIR, "sim4-coverage.pdf"))
    println("  Saved sim4-coverage.pdf\n")
end

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

function main()
    println("\n" * "=" ^ 72)
    println("  uHGF Paper — All figures")
    println("  Output directory: $FIGDIR")
    println("=" ^ 72 * "\n")

    println("── Theory figures ──────────────────────────────────────────────────")
    figure_origapprox()
    figure_Jcomponents()
    figure_bimodal()

    println("\n── Simulation figures ──────────────────────────────────────────────")
    run_simulation_1()
    run_simulation_2()
    run_simulation_3()
    run_simulation_4()

    println("\n" * "=" ^ 72)
    println("  All figures complete.")
    println("=" ^ 72)
end

main()
