### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ ec000001-0001-4001-8001-000000000001
begin
	import Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
	using Plots
	using Printf
	using Statistics
end

# ╔═╡ ec000001-0001-4001-8001-000000000002
md"# Effective coupling rate for multi-child volatility updates

## Background

In the standard two-level HGF, a single volatility parent $x_j$ has one child $x_{j-1}$
and the volatility update is driven by the weighted prediction error $\delta_{j-1}$
of that child.  When the parent has **multiple children** $x_{j-1}^{(1)}, \ldots, x_{j-1}^{(n)}$
each with its own coupling $\kappa^{(i)}$ and $\omega^{(i)}$, the variational energy
for the parent becomes a sum of per-child terms, and the mode equation can no longer
be solved by a single Lambert W₀ evaluation.

The **effective coupling rate** idea replaces that sum of exponentials with a single
effective exponential whose coupling rate $\kappa_e$ is the weighted average of the
children's coupling rates.  This lets us reuse the single-child Lambert W₀ formula
essentially unchanged, at the cost of approximating the aggregate curvature rather than
computing it exactly.

The natural baseline for comparison is the **per-child uHGF**: apply the paper's
single-child Lambert W₀ solve to each child independently, yielding $n+1$ expansion
points (one at $\hat\mu$ plus one per child).  Each child's secondary mode is found in
isolation; the $n+1$ Gaussians are then blended via a softmax over the full multi-child
variational energy.
"

# ╔═╡ ec000001-0001-4001-8001-000000000003
md"## Derivation

### Multi-child variational energy

With $n$ children, the volatility parent's variational energy (in the canonical variable
$x = \mu_j$) is

$$I(x) = -\frac{\hat\pi}{2}(x - \hat\mu)^2
       + \sum_{i=1}^{n} \left[
           -\frac{1}{2}\log\!\left(\alpha_i + t\,e^{\kappa^{(i)} x + \omega^{(i)}}\right)
           - \frac{\beta_i}{2(\alpha_i + t\,e^{\kappa^{(i)} x + \omega^{(i)}})}
         \right]$$

where $\hat\pi = 1/\hat\sigma_j$ is the prior precision of the parent,
$\alpha_i = \sigma_{j-1}^{0,(i)}$ is child $i$'s previous variance,
and $\beta_i = 1/\pi_{j-1}^{(i)} + (\mu_{j-1}^{(i)} - \hat\mu_{j-1}^{(i)})^2$
is the total posterior uncertainty of child $i$.

Setting $I'(x)=0$ gives the mode equation

$$\hat\pi(\hat\mu - x)
  = \sum_{i=1}^n \frac{\kappa^{(i)}}{2} w^{(i)}(x)\,\delta^{(i)}(x)$$

with $w^{(i)}(x) = t e^{\kappa^{(i)} x + \omega^{(i)}} /
(\alpha_i + t e^{\kappa^{(i)} x + \omega^{(i)}})$
and $\delta^{(i)}(x) = \beta_i / (\alpha_i + t e^{\kappa^{(i)} x + \omega^{(i)}}) - 1$.

This has no closed form when the $\kappa^{(i)}$ differ.
"

# ╔═╡ ec000001-0001-4001-8001-000000000004
md"### Effective coupling rate approximation

**Key observation.** In the limit $\alpha_i \to 0$, each per-child term simplifies because
$w^{(i)} \to 1$ and $\beta_i / (\alpha_i + t e^{\kappa^{(i)} x + \omega^{(i)}}) \approx
\beta_i / (t e^{\kappa^{(i)} x + \omega^{(i)}})$.
The mode equation then becomes

$$\hat\pi(\hat\mu - x)
  \approx \frac{1}{2}\sum_i \kappa^{(i)} \frac{\beta_i}{t\,e^{\kappa^{(i)} x + \omega^{(i)}}} - \frac{1}{2}\sum_i \kappa^{(i)}.$$

Defining $\Omega_i = t\,e^{\omega^{(i)}}$ and substituting $x = \hat\mu + \Delta$,
the dominant $x$-dependence of the right-hand side is through the exponentials
$e^{-\kappa^{(i)} \Delta}$.  We approximate these by a **single effective exponential**
$e^{-\kappa_e \Delta}$ via a moment-matching argument on the log-Laplace exponents.

Define the zeroth and first coupling moments

$$S_0 = \sum_i \frac{\kappa^{(i)} \beta_i}{\Omega_i}, \qquad
S_1 = \sum_i \frac{(\kappa^{(i)})^2 \beta_i}{\Omega_i},$$

and the **effective coupling rate**

$$\kappa_e := \frac{S_1}{S_0}.$$

This is the $\beta_i / \Omega_i$-weighted average of $\kappa^{(i)}$; it is exact when
all couplings are equal.

Substituting, the aggregate mode equation (in the effective canonical variable
$y = \kappa_e x + \log t + \omega_e$, with $\omega_e = \log(S_0 / \kappa_e)$) reduces
to the same form as the single-child equation, which is solved by

$$y^* = \hat\gamma_e + v - \frac{1}{2\hat\pi_y},$$

where $\hat\gamma_e = \log t + \kappa_e \hat\mu + \omega_e$,
$\hat\pi_y = \hat\pi / \kappa_e^2$, and $v = W_0\!\left(\frac{S_0}{2\hat\pi_y}\,e^{1/(2\hat\pi_y) - \hat\gamma_e}\right)$.

Returning to $x$-space: $x^* = (y^* - \log t - \omega_e) / \kappa_e$.
"

# ╔═╡ ec000001-0001-4001-8001-000000000005
md"### Algorithm

Given a parent node with prior $(\hat\mu, \hat\pi)$ and $n$ children, each with
quantities $(\kappa^{(i)}, \omega^{(i)}, \alpha_i, \beta_i)$:

1. **Compute moments**
$$S_0 = \sum_i \frac{\kappa^{(i)} \beta_i}{\Omega_i}, \quad
S_1 = \sum_i \frac{(\kappa^{(i)})^2 \beta_i}{\Omega_i}, \quad \Omega_i = t\,e^{\omega^{(i)}}$$

2. **Effective coupling and offset**
$$\kappa_e = S_1/S_0, \quad \omega_e = \log(S_0/\kappa_e)$$

3. **Lambert W₀ solve** (as in the single-child uHGF)
$$\hat\gamma_e = \log t + \kappa_e \hat\mu + \omega_e, \quad
\hat\pi_y = \hat\pi/\kappa_e^2, \quad
x^* = \hat\mu - \frac{1}{2\hat\pi_y\kappa_e} + \frac{1}{\kappa_e}W_0\!\left(\frac{\kappa_e S_0}{2\hat\pi}\,e^{\kappa_e/(2\hat\pi)\cdot\kappa_e - \hat\gamma_e}\right)$$

4. **Quadratic expansion at $x^*$** to obtain precision $\pi_2$ and mean $\mu_2$, using
   the true multi-child energy (not the effective approximation).

5. **Blend** with expansion 1 (at $\hat\mu$) using softmax weight from $I(\mu_1)$ vs $I(x^*)$.

6. **Moment-match** the two-component Gaussian mixture.

The key cost over the single-child case is the one-time computation of $S_0, S_1$ —
both $O(n)$.  The Lambert W₀ solve remains $O(1)$.
"

# ╔═╡ ec000001-0001-4001-8001-000000000006
md"## Numerical illustration

We construct a two-child example and compare:
- **Per-child uHGF** (paper's algorithm applied per child: $n+1$ expansion points,
  one Lambert W₀ solve per child, blend via softmax over full multi-child energy)
- **Effective coupling** (one effective Lambert W₀ solve, then full-energy expansion;
  $O(1)$ Lambert W₀ evaluations vs $O(n)$ for per-child)
"

# ╔═╡ ec000001-0001-4001-8001-000000000007
begin
	# Lambert W₀ via Halley's method
	function lambert_w0(z::Float64)
		z < 0.0   && return NaN
		z < 1e-10 && return z
		w = z > 3.0 ? log(z) - log(log(z)) : 1.0
		for _ in 1:10
			ew = exp(w)
			f  = w * ew - z
			fp = ew * (1.0 + w)
			w -= f / (fp - f * (1.0 + w) / (2.0 * (1.0 + w)))
		end
		return w
	end

	# Variational energy for a multi-child parent
	# κs, ωs, αs, βs are length-n vectors; pihat, muhat are scalars; t is step size
	function I_multi(x, pihat, muhat, κs, ωs, αs, βs, t)
		energy = -0.5 * pihat * (x - muhat)^2
		for i in eachindex(κs)
			denom = αs[i] + t * exp(κs[i] * x + ωs[i])
			energy += -0.5 * log(denom) - 0.5 * βs[i] / denom
		end
		return energy
	end

	# Quadratic expansion of I_multi at expansion point x0
	# Returns (precision, mean) of the Gaussian approximation
	function laplace_at(x0, pihat, muhat, κs, ωs, αs, βs, t)
		# Second derivative of I at x0 (precision = -I'')
		ddI = -pihat
		for i in eachindex(κs)
			s = t * exp(κs[i] * x0 + ωs[i])
			denom = αs[i] + s
			w = s / denom
			δ = βs[i] / denom - 1.0
			ddI += -0.5 * κs[i]^2 * (w * (w + (2w - 1) * δ))
		end
		π_approx = -ddI
		if π_approx <= 0.0
			# Fall back to concave-only second derivative (guaranteed negative)
			ddI_safe = -pihat
			for i in eachindex(κs)
				s = t * exp(κs[i] * x0 + ωs[i])
				w = s / (αs[i] + s)
				ddI_safe += -0.5 * κs[i]^2 * w * (1.0 - w)
			end
			π_approx = -ddI_safe
		end
		# First derivative of I at x0
		dI = -pihat * (x0 - muhat)
		for i in eachindex(κs)
			s = t * exp(κs[i] * x0 + ωs[i])
			denom = αs[i] + s
			w = s / denom
			δ = βs[i] / denom - 1.0
			dI += 0.5 * κs[i] * w * δ
		end
		μ_approx = x0 - dI / (-π_approx)
		return π_approx, μ_approx
	end

	# Per-child uHGF: paper's single-child Lambert W₀ applied to each child independently.
	# Returns n+1 expansion points: one at muhat, one per child.
	function per_child_mode(i, pihat, muhat, κs, ωs, αs, βs, t)
		γ_c   = log(t) + κs[i] * muhat + ωs[i]
		pihat_y = pihat / κs[i]^2
		log_W_arg = log(βs[i] / (2.0 * pihat_y)) + 0.5 / pihat_y - γ_c
		W_arg = isfinite(log_W_arg) ? exp(min(log_W_arg, 709.0)) : NaN
		v = lambert_w0(W_arg)
		y_star = γ_c + v - 0.5 / pihat_y
		x_star = (y_star - log(t) - ωs[i]) / κs[i]
		return x_star
	end

	function uhgf_per_child(pihat, muhat, κs, ωs, αs, βs, t)
		n = length(κs)
		# Gather all expansion points and their Laplace approximations
		points = [muhat; [per_child_mode(i, pihat, muhat, κs, ωs, αs, βs, t) for i in 1:n]]
		πs = Float64[]
		μs = Float64[]
		Is = Float64[]
		for xp in points
			if isfinite(xp)
				πp, μp = laplace_at(xp, pihat, muhat, κs, ωs, αs, βs, t)
				Ip = I_multi(μp, pihat, muhat, κs, ωs, αs, βs, t)
				push!(πs, πp); push!(μs, μp); push!(Is, Ip)
			end
		end
		isempty(πs) && return NaN, NaN
		# Softmax weights from variational energy
		Is_shifted = Is .- maximum(Is)
		ws = exp.(Is_shifted); ws ./= sum(ws)
		# Gaussian mixture moment matching
		μ  = sum(ws .* μs)
		σ² = sum(ws .* (1.0 ./ πs)) + sum(ws .* (μs .- μ).^2)
		return 1.0/σ², μ
	end

	# Effective coupling rate mode-finder
	function effective_coupling_mode(pihat, muhat, κs, ωs, βs, t)
		S0 = sum(κs[i] * βs[i] / (t * exp(ωs[i])) for i in eachindex(κs))
		S1 = sum(κs[i]^2 * βs[i] / (t * exp(ωs[i])) for i in eachindex(κs))
		κe = S1 / S0
		ωe = log(S0 / κe)
		γ_hat = log(t) + κe * muhat + ωe
		pihat_y = pihat / κe^2
		log_W_arg = log(κe * S0 / (2.0 * pihat)) + 0.5 / pihat_y - γ_hat
		W_arg = isfinite(log_W_arg) ? exp(min(log_W_arg, 709.0)) : NaN
		v = lambert_w0(W_arg)
		y_star = γ_hat + v - 0.5 / pihat_y
		x_star = (y_star - log(t) - ωe) / κe
		return x_star
	end

	# Full uHGF effective coupling update
	function uhgf_effective(pihat, muhat, κs, ωs, αs, βs, t)
		# Expansion 1: at prior mean
		π1, μ1 = laplace_at(muhat, pihat, muhat, κs, ωs, αs, βs, t)

		# Expansion 2: at effective coupling mode
		x_star = effective_coupling_mode(pihat, muhat, κs, ωs, βs, t)
		if isfinite(x_star)
			π2, μ2 = laplace_at(x_star, pihat, muhat, κs, ωs, αs, βs, t)
		else
			π2, μ2 = π1, μ1
		end

		# Softmax blend
		I1 = I_multi(μ1, pihat, muhat, κs, ωs, αs, βs, t)
		I2 = I_multi(μ2, pihat, muhat, κs, ωs, αs, βs, t)
		d  = I1 - I2
		b  = isnan(d) ? 0.5 : (d >= 0 ? exp(-d)/(1+exp(-d)) : 1/(1+exp(d)))

		# Mixture moment matching
		μ   = (1 - b)*μ1 + b*μ2
		σ²  = (1 - b)/π1 + b/π2 + b*(1 - b)*(μ1 - μ2)^2
		return 1/σ², μ
	end

	println("Helper functions defined.")
end

# ╔═╡ ec000001-0001-4001-8001-000000000008
md"### Example: Two-child node with contrasting couplings

We use $n=2$ children with $\kappa^{(1)} = 1$, $\kappa^{(2)} = 2$ and sweep the
parent prior mean $\hat\mu \in [-15, 5]$, holding all other parameters fixed.
This sweep forces the variational energy through both concave and convex regimes,
exercising the blending mechanism.

The per-child uHGF uses two Lambert W₀ evaluations (one per child) and produces
$n+1 = 3$ expansion points.  The effective coupling uses one Lambert W₀ evaluation
(plus $O(n)$ arithmetic for $S_0, S_1$) and produces 2 expansion points.
"

# ╔═╡ ec000001-0001-4001-8001-000000000009
begin
	# Fixed parameters
	pihat_ex = 1.0       # parent prior precision
	κs_ex  = [1.0, 2.0]
	ωs_ex  = [0.0, 0.0]
	αs_ex  = [0.005, 0.005]   # small prior variance → large PEs expected
	βs_ex  = [0.25, 0.25]     # moderate posterior uncertainty
	t_ex   = 1.0

	muhat_grid = range(-15.0, 5.0; length=200)

	π_pc_grid  = Float64[]
	μ_pc_grid  = Float64[]
	π_eff_grid = Float64[]
	μ_eff_grid = Float64[]

	for muhat in muhat_grid
		πp, μp = uhgf_per_child(pihat_ex, muhat, κs_ex, ωs_ex, αs_ex, βs_ex, t_ex)
		push!(π_pc_grid, πp)
		push!(μ_pc_grid, μp)
		πe, μe = uhgf_effective(pihat_ex, muhat, κs_ex, ωs_ex, αs_ex, βs_ex, t_ex)
		push!(π_eff_grid, πe)
		push!(μ_eff_grid, μe)
	end

	@printf "Per-child uHGF failures:   %d / %d\n" sum(isnan.(π_pc_grid)) length(muhat_grid)
	@printf "Effective coupling failures: %d / %d\n" sum(isnan.(π_eff_grid)) length(muhat_grid)
end

# ╔═╡ ec000001-0001-4001-8001-00000000000a
begin
	p1 = plot(muhat_grid, π_pc_grid;
		label="Per-child uHGF", color=:steelblue, linewidth=2,
		xlabel="μ̂ (parent prior mean)", ylabel="Posterior precision π",
		title="Posterior precision across μ̂ sweep",
		titlefontsize=9, legend=:topright)
	plot!(p1, muhat_grid, π_eff_grid;
		label="Effective coupling", color=:crimson, linewidth=2)
	hline!(p1, [0.0]; color=:black, linestyle=:dash, linewidth=1, label="π = 0")

	p2 = plot(muhat_grid, replace(μ_pc_grid, NaN => missing);
		label="Per-child uHGF", color=:steelblue, linewidth=2,
		xlabel="μ̂ (parent prior mean)", ylabel="Posterior mean μ",
		title="Posterior mean across μ̂ sweep",
		titlefontsize=9, legend=:topright)
	plot!(p2, muhat_grid, μ_eff_grid;
		label="Effective coupling", color=:crimson, linewidth=2)

	plot(p1, p2; layout=(2,1), size=(700, 500),
		plot_title="Two-child volatility parent  (κ₁=1, κ₂=2, α=0.005, β=0.25)",
		plot_titlefontsize=10, plot_titlevspan=0.06,
		margin=4Plots.mm, left_margin=8Plots.mm)
end

# ╔═╡ ec000001-0001-4001-8001-00000000000b
md"### KL divergence vs. true variational posterior

We compute the KL divergence between the true (numerically integrated) variational
posterior and each approximation at every grid point.  The true posterior is obtained
by numerical integration on a grid of 6 001 points over $[-60, 60]$.
"

# ╔═╡ ec000001-0001-4001-8001-00000000000c
begin
	xs_kl = collect(range(-60.0, 60.0; length=6001))
	dx_kl = xs_kl[2] - xs_kl[1]

	gauss_pdf(x, μ, π_p) = sqrt(π_p / (2π)) * exp(-0.5 * π_p * (x - μ)^2)

	function kl_gauss(xs, dx, I_fn, μ_q, π_q)
		π_q <= 0.0 && return NaN
		p_unnorm = [exp(I_fn(x)) for x in xs]
		Z = sum(p_unnorm) * dx
		kl = 0.0
		for (i, x) in enumerate(xs)
			p = p_unnorm[i] / Z
			p < 1e-300 && continue
			q = gauss_pdf(x, μ_q, π_q)
			kl += p * (q > 1e-300 ? log(p/q) : 50.0) * dx
		end
		return max(kl, 0.0)
	end

	kl_pc_vec  = Float64[]
	kl_eff_vec = Float64[]

	for (k, muhat) in enumerate(muhat_grid)
		I_fn = x -> I_multi(x, pihat_ex, muhat, κs_ex, ωs_ex, αs_ex, βs_ex, t_ex)
		kl_p = kl_gauss(xs_kl, dx_kl, I_fn, μ_pc_grid[k],  π_pc_grid[k])
		kl_e = kl_gauss(xs_kl, dx_kl, I_fn, μ_eff_grid[k], π_eff_grid[k])
		push!(kl_pc_vec, kl_p)
		push!(kl_eff_vec, kl_e)
	end

	valid_p = filter(isfinite, kl_pc_vec)
	valid_e = filter(isfinite, kl_eff_vec)
	@printf "Per-child uHGF — mean KL: %.4f,  max KL: %.4f,  failures: %d\n" mean(valid_p) maximum(valid_p) sum(isnan.(kl_pc_vec))
	@printf "Eff. coupling  — mean KL: %.4f,  max KL: %.4f,  failures: %d\n" mean(valid_e) maximum(valid_e) sum(isnan.(kl_eff_vec))
end

# ╔═╡ ec000001-0001-4001-8001-00000000000d
begin
	kl_pc_plot  = replace(kl_pc_vec,  NaN => missing)
	kl_eff_plot = replace(kl_eff_vec, NaN => missing)

	plot(muhat_grid, kl_pc_plot;
		label="Per-child uHGF", color=:steelblue, linewidth=2,
		xlabel="μ̂ (parent prior mean)", ylabel="KL divergence  KL(p ‖ q)",
		title="Approximation quality: KL divergence  (two-child parent)",
		titlefontsize=9, legend=:topright,
		size=(700, 350), margin=4Plots.mm, left_margin=8Plots.mm)
	plot!(muhat_grid, kl_eff_plot;
		label="Effective coupling", color=:crimson, linewidth=2)
end

# ╔═╡ ec000001-0001-4001-8001-00000000000e
md"## Properties and limitations

### Exact cases
- **$n = 1$**: the effective coupling formula reproduces the single-child Lambert W₀ solve exactly, so it is identical to the paper's uHGF.
- **All $\kappa^{(i)}$ equal**: $\kappa_e = \kappa$, $\Omega_e = S_0/\kappa$, and the effective formula collapses to the single-child case with the aggregate $\beta_{\text{eff}} = \sum_i \beta_i$.

### Approximation error vs. per-child
The effective coupling finds a single aggregate secondary mode.  When children have
very different $\kappa^{(i)}$, each child can pull the secondary mode to a distinct
location — these are not captured by a single $x^*$.  The per-child approach
($n+1$ expansion points, one Lambert W₀ per child) is the principled extension of
the paper's algorithm and will generally outperform the effective coupling when the
$\kappa^{(i)}$ are heterogeneous.

### Cost comparison
| Method | Lambert W₀ solves | Expansion points |
|--------|------------------|------------------|
| Per-child uHGF | $n$ | $n + 1$ |
| Effective coupling, one-shot | $1$ | $2$ |
| Effective coupling, iterated | $1 \times k$ iterations | $2$ |

### Iterative refinement
After finding $x^*$, one can recompute $S_0, S_1$ evaluated at $x^*$ rather than at
$\hat\mu$, yielding an updated $\kappa_e(x^*)$ and a tighter mode estimate.  This
iterated version converges in two to three steps for typical parameters.
"

# ╔═╡ ec000001-0001-4001-8001-00000000000f
md"## Iterative refinement

We re-evaluate $S_0$, $S_1$ at the current estimate $x^*$ and iterate the Lambert W₀
solve up to a fixed number of steps.
"

# ╔═╡ ec000001-0001-4001-8001-000000000010
begin
	function effective_coupling_mode_iterated(pihat, muhat, κs, ωs, αs, βs, t;
	                                           n_iter=3)
		x = muhat
		for _ in 1:n_iter
			# Re-evaluate S0, S1 at current x (using actual denominators, not α→0 approx)
			S0 = sum(κs[i] * βs[i] / (αs[i] + t*exp(κs[i]*x + ωs[i]))
			         for i in eachindex(κs))
			S1 = sum(κs[i]^2 * βs[i] / (αs[i] + t*exp(κs[i]*x + ωs[i]))
			         for i in eachindex(κs))
			S0 <= 0 && break
			κe = S1 / S0
			κe <= 0 && break
			ωe = log(S0 / κe)
			γ_hat = log(t) + κe * muhat + ωe
			pihat_y = pihat / κe^2
			log_W_arg = log(κe * S0 / (2.0 * pihat)) + 0.5/pihat_y - γ_hat
			W_arg = isfinite(log_W_arg) ? exp(min(log_W_arg, 709.0)) : NaN
			v = lambert_w0(W_arg)
			y_star = γ_hat + v - 0.5 / pihat_y
			x_new = (y_star - log(t) - ωe) / κe
			!isfinite(x_new) && break
			x = x_new
		end
		return x
	end

	function uhgf_effective_iterated(pihat, muhat, κs, ωs, αs, βs, t; n_iter=3)
		π1, μ1 = laplace_at(muhat, pihat, muhat, κs, ωs, αs, βs, t)
		x_star = effective_coupling_mode_iterated(pihat, muhat, κs, ωs, αs, βs, t;
		                                           n_iter=n_iter)
		if isfinite(x_star)
			π2, μ2 = laplace_at(x_star, pihat, muhat, κs, ωs, αs, βs, t)
		else
			π2, μ2 = π1, μ1
		end
		I1 = I_multi(μ1, pihat, muhat, κs, ωs, αs, βs, t)
		I2 = I_multi(μ2, pihat, muhat, κs, ωs, αs, βs, t)
		d  = I1 - I2
		b  = isnan(d) ? 0.5 : (d >= 0 ? exp(-d)/(1+exp(-d)) : 1/(1+exp(d)))
		μ  = (1-b)*μ1 + b*μ2
		σ² = (1-b)/π1 + b/π2 + b*(1-b)*(μ1-μ2)^2
		return 1/σ², μ
	end

	# Compare one-shot vs. iterated on the same sweep
	kl_iter_vec = Float64[]
	for (k, muhat) in enumerate(muhat_grid)
		πi, μi = uhgf_effective_iterated(pihat_ex, muhat, κs_ex, ωs_ex,
		                                  αs_ex, βs_ex, t_ex)
		I_fn = x -> I_multi(x, pihat_ex, muhat, κs_ex, ωs_ex, αs_ex, βs_ex, t_ex)
		kl_i = kl_gauss(xs_kl, dx_kl, I_fn, μi, πi)
		push!(kl_iter_vec, kl_i)
	end

	ve = filter(isfinite, kl_eff_vec)
	vi = filter(isfinite, kl_iter_vec)
	@printf "One-shot eff. — mean KL: %.4f,  max KL: %.4f\n" mean(ve) maximum(ve)
	@printf "Iterated eff. — mean KL: %.4f,  max KL: %.4f\n" mean(vi) maximum(vi)
end

# ╔═╡ ec000001-0001-4001-8001-000000000011
begin
	kl_iter_plot = replace(kl_iter_vec, NaN => missing)

	plot(muhat_grid, kl_pc_plot;
		label="Per-child uHGF", color=:steelblue, linewidth=2,
		xlabel="μ̂ (parent prior mean)", ylabel="KL divergence  KL(p ‖ q)",
		title="Per-child uHGF vs. one-shot and iterated effective coupling",
		titlefontsize=9, legend=:topright,
		size=(700, 350), margin=4Plots.mm, left_margin=8Plots.mm)
	plot!(muhat_grid, kl_eff_plot;
		label="Effective coupling (one-shot)", color=:crimson, linewidth=2,
		linestyle=:dash)
	plot!(muhat_grid, kl_iter_plot;
		label="Effective coupling (iterated)", color=:crimson, linewidth=2)
end

# ╔═╡ ec000001-0001-4001-8001-000000000012
begin
	vp   = filter(isfinite, kl_pc_vec)
	ve2  = filter(isfinite, kl_eff_vec)
	vi2  = filter(isfinite, kl_iter_vec)
	fp   = sum(isnan.(kl_pc_vec))
	fe   = sum(isnan.(kl_eff_vec))
	fi   = sum(isnan.(kl_iter_vec))

	Markdown.parse("""
## Summary

| Method | Lambert W₀ solves | Mean KL | Max KL | Failures |
|--------|-------------------|---------|--------|----------|
| Per-child uHGF | ``n`` | $(round(mean(vp); digits=4)) | $(round(maximum(vp); digits=4)) | $(fp) |
| Eff. coupling, one-shot | ``1`` | $(round(mean(ve2); digits=4)) | $(round(maximum(ve2); digits=4)) | $(fe) |
| Eff. coupling, iterated | ``k`` | $(round(mean(vi2); digits=4)) | $(round(maximum(vi2); digits=4)) | $(fi) |

The effective coupling rate is a low-cost route to secondary expansion for multi-child
nodes.  It is exact for equal couplings (``n=1`` or all ``\\kappa^{(i)}`` equal) and
provides a good approximation for moderate heterogeneity.  For strongly heterogeneous
couplings the per-child approach is more accurate but costs ``O(n)`` Lambert W₀ solves.
The iterated effective coupling closes much of this gap at ``O(1)`` Lambert W₀ cost.
""")
end

# ╔═╡ Cell order:
# ╟─ec000001-0001-4001-8001-000000000002
# ╠═ec000001-0001-4001-8001-000000000001
# ╟─ec000001-0001-4001-8001-000000000003
# ╟─ec000001-0001-4001-8001-000000000004
# ╟─ec000001-0001-4001-8001-000000000005
# ╟─ec000001-0001-4001-8001-000000000006
# ╠═ec000001-0001-4001-8001-000000000007
# ╟─ec000001-0001-4001-8001-000000000008
# ╠═ec000001-0001-4001-8001-000000000009
# ╠═ec000001-0001-4001-8001-00000000000a
# ╟─ec000001-0001-4001-8001-00000000000b
# ╠═ec000001-0001-4001-8001-00000000000c
# ╠═ec000001-0001-4001-8001-00000000000d
# ╟─ec000001-0001-4001-8001-00000000000e
# ╟─ec000001-0001-4001-8001-00000000000f
# ╠═ec000001-0001-4001-8001-000000000010
# ╠═ec000001-0001-4001-8001-000000000011
# ╠═ec000001-0001-4001-8001-000000000012
