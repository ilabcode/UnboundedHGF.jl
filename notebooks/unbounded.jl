### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ d0173c55-4f11-4856-8756-d9dd9db8ca3b
begin
	import Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
	using Plots
	using PlutoUI
	using QuadGK
	using LaTeXStrings
	using DelimitedFiles
	using Printf
	using Statistics: mean
	using UnboundedHGF
end

# ╔═╡ 0b6a202a-1a1c-11ef-38b9-cba623dfd57d
md"# Stable HGF updates for arbitrarily large prediction errors"

# ╔═╡ 02bb8240-a363-413b-a74f-864837038051
md"## Inference on node values under the mean field approximation

Under the mean field approximation, the posterior probability distribution $q$ on the value of a node $x$ in a network is the normalized exponential of the node's variational energy $I$ (for details see Mathys, 2012, Appendix B):

$$q(x) = \frac{1}{\mathcal{Z}} \exp(I(x)) \quad \text{with } \mathcal{Z} := \int_{-\infty}^{\infty} \exp(I(x)) \mathrm{d}x$$

This means that the approximate posterior $q$ is Gaussian if and only if the variational energy $I$ is quadratic."

# ╔═╡ e908b646-1c22-49c5-983b-c6c6174babe1
md"## Variational energy function of volatility coupling

An HGF state node $x$ which is volatility parent of $x_\dagger$ and in turn has volatility parent $x_\ddagger$ has the following variational energy (Mathys et al., 2011; 2014), where $t$ denotes the time elapsed since the previous update: 

$$\begin{aligned}
I(x) =& - \frac{1}{2} \log \left( \sigma_\dagger^0 + t\exp(\kappa_\dagger x + \omega_\dagger) \right)\\ &- \frac{1}{2} \frac{\sigma_\dagger + \left( \mu_\dagger - \hat{\mu}_\dagger \right)^2}{\sigma_\dagger^0 + t\exp(\kappa_\dagger x + \omega_\dagger)} \\&- \frac{1}{2} \frac{1}{\sigma^0 + t\exp(\kappa \mu_\ddagger^0 + \omega)} \left( x - \hat{\mu} \right)^2,
\end{aligned}$$
"

# ╔═╡ 2c657235-6d9f-4c71-bf6b-9605d0fd8168
md"where the various parameters signify the following:

-  $t$: Time elapsed since the previous update
-  $\sigma_\dagger^0$: Previous $\sigma$ of child
-  $\kappa_\dagger$: $\kappa$ of child
-  $\omega_\dagger$: $\omega$ of child
-  $\sigma_\dagger$: $\sigma$ of child
-  $\mu_\dagger$: $\mu$ of child
-  $\hat{\mu}_\dagger$: Predicted $\mu$ of child
-  $\sigma^0$: Previous $\sigma$
-  $\kappa$: $\kappa$
-  $\omega$: $\omega$
-  $\mu_\ddagger^0$: Previous $\mu$ of parent
-  $\hat{\mu}$: Predicted $\mu$"

# ╔═╡ c0d483b4-73d5-4904-a340-56c2d467dfba
function I(x, t, σc0, κc, ωc, σc, μc, μc0, σ0, κ, ω, μp0, μ0)
	- 0.5*log(σc0 + t*exp(κc*x + ωc)) - 0.5*(σc + (μc - μc0)^2)/(σc0 + t*exp(κc*x + ωc)) - 0.5/(σ0 + t*exp(κ*μp0 + ω))*(x - μ0)^2
end

# ╔═╡ 5cccfacd-24a4-4085-80e0-49c06e8aaef8
md"## Canonical parameterization of the variational energy

For our current purposes, it is useful to introduce a particular parameterization of the variational energy that uses only three parameters yet retains all properties we are concerned with here. We call this the *canonical variational energy function* $J$.

We obtain it by setting $t$ to one and letting all $\mu$'s and $\omega$'s be zero and letting all $\sigma$'s and $\kappa$'s be one, with three exceptions. First, $\sigma_\dagger^0$ remains as a parameter and will be written $\alpha$ for simplicity. Second, the posterior total uncertainty about the child $\sigma_\dagger + \left( \mu_\dagger - \hat{\mu}_\dagger \right)^2$ is taken as a single parameter $\beta$. Finally, $\hat{\mu}$ remains and is called $\gamma$.

$$J(x) := - \frac{1}{2} \log \left( \alpha + e^x \right) - \frac{1}{2} \frac{\beta}{\alpha + e^x} - \frac{1}{4} (x - \gamma)^2 \quad \text{with } \alpha, \beta > 0, \gamma \in \mathbb{R}$$
"

# ╔═╡ 040f42f6-cbd4-4028-a773-50188c4888f0
J(x, α, β, γ) = - 0.5 * log(α + exp(x)) - 0.5 * β / (α + exp(x)) - 0.25 * (x - γ)^2

# ╔═╡ 68373d5c-4b35-4050-b3fa-39cb66203ccb
md"## Quadratic approximation in the original HGF formulation

Since $J$ is not quadratic, the approximate posterior implied by it is not Gaussian. To obtain a Gaussian approximate posterior, we therefore need to find a quadratic approximation to $J$.

In the original HGF formulation (Mathys et al., 2011), this is achieved by expanding $J$ to second order at $\gamma$.

$$\hat{J}(x) := J(\gamma) + J'(\gamma) (x - \gamma) + \frac{1}{2} J''(\gamma) (x - \gamma)^2.$$

This yields a Gaussian posterior with mean $\mu_J$ and precision (inverse variance) $\pi_J$:

$$q(x) = \sqrt{\frac{\pi_J}{2 \pi}} \exp \left(- \frac{\pi_J}{2} \left(x - \mu_J \right)^2 \right)$$

In the calculation of $\mu_J$ and $\pi_J$, we need the first and second derivatives, $J'$ and $J''$, of $J$:

$$\begin{align}
J'(x) =& \frac{1}{2} \frac{e^x}{\alpha + e^x} \left( \frac{\beta}{\alpha + e^x} - 1 \right) - \frac{1}{2} (x - \gamma) \\[2em]
J''(x) =& - \frac{1}{2} \frac{e^x}{\alpha + e^x} \left( \frac{\alpha}{\alpha + e^x} + \frac{(e^x - \alpha) \beta}{(\alpha + e^x)^2} \right) - \frac{1}{2} \\ =& - \frac{1}{2} \frac{e^x}{\alpha + e^x} \left( \frac{e^x}{\alpha + e^x} + \frac{e^x - \alpha}{\alpha + e^x} \left( \frac{\beta}{\alpha + e^x} - 1 \right) \right) - \frac{1}{2}.
\end{align}$$"

# ╔═╡ 2185648d-57b5-4d5f-a1f1-1bb04666b64d
md"We simplify notation with the following definitions:

$$\begin{align}
w(x) :=& \frac{e^x}{\alpha + e^x} \\[2em]
\delta(x) :=& \frac{\beta}{\alpha + e^x} - 1
\end{align}$$"

# ╔═╡ 7f22a754-d136-4572-b778-108971478179
w(α, x) = exp(x) / (α + exp(x))

# ╔═╡ eac83e66-13e0-4405-8bca-b70e19fbf208
δ(α, β, x) = β / (α + exp(x)) - 1

# ╔═╡ f42b26c0-9968-4a79-83c1-5327240f9d8b
md"With

$$\begin{align}
w'(x) =& w(x)(1 - w(x)) \\[2em]
\delta'(x) =& -w(x)(\delta(x) + 1),
\end{align}$$

this gives us

$$\begin{align}
J'(x) =& \frac{1}{2} w(x) \delta(x) - \frac{1}{2} (x - \gamma) \\[2em]
J''(x) =& - \frac{1}{2} w(x) \left( w(x) + (2w(x) - 1) \delta(x) \right) - \frac{1}{2}.
\end{align}$$"

# ╔═╡ 9a70fa82-8b3c-4c3c-9950-57c1a98a19ad
dJ(x, α, β, γ) = 0.5 * w(α, x) * δ(α, β, x) - 0.5 * (x - γ)

# ╔═╡ 89eb7f34-e905-4463-a5b4-25ad70e65648
ddJ(x, α, β) = - 0.5 * w(α, x) * (w(α, x) + (2 * w(α, x) - 1) * δ(α, β, x)) - 0.5

# ╔═╡ 6c3777f6-2ff9-4dcf-8e37-83f18985381f
md"With this, we can determine the sufficient statistics of the Gaussian approximate posterior:

$$\begin{align}
\pi_J =& - J''(\gamma) = \frac{1}{2} w(\gamma) \left( w(\gamma) + (2w(\gamma) - 1) \delta(γ) \right) + \frac{1}{2} \\[2em]
\mu_J =& \gamma - \frac{J'(\gamma)}{J''(\gamma)} = \gamma + \frac{w(\gamma)}{2 \pi_J} \delta(\gamma).
\end{align}$$"

# ╔═╡ 2fad2f32-fa62-459c-98e4-40e9d116ea5c
πJ(α, β, γ) = 0.5 * w(α, γ) * (w(α, γ) + (2*w(α, γ) - 1) * δ(α, β, γ)) + 0.5

# ╔═╡ 4aca9637-1694-4f5c-bef2-304b8b7b49df
μJ(α, β, γ) = γ + 0.5 / πJ(α, β, γ) * w(α, γ) * δ(α, β, γ)

# ╔═╡ cd768b8b-84f6-4cd0-973b-e60ae8e9835a
md"The following interactive visualization of $\hat{J}$ illustrates how the original quadratic approximation works. By changing $\alpha$ and $\beta$ we can create a convex region in the variational energy $J$. If we then move $\gamma$ to the convex region, we make the curvature of $\hat{J}$ positive, leading to negative posterior precision."

# ╔═╡ e0a70b3f-edf3-45dc-b672-a067d84895f0
Jhat(x, α, β, γ) = J(γ, α, β, γ) + dJ(γ, α, β, γ) * (x - γ) + 0.5 * ddJ(γ, α, β) *(x - γ)^2

# ╔═╡ 92243446-65a4-4787-8a73-f1917083c1be
md"#### Parameter values"

# ╔═╡ ee860cab-e097-46b2-9bd9-1f7a2de45b36
md"##### $\alpha$"

# ╔═╡ 4bc193ac-12eb-4889-ad9d-0bb47ab58e8d
@bind α Slider(0.005:0.005:1, default = 1.0)

# ╔═╡ fb06391c-c460-4732-9cea-87e22ffdd665
α

# ╔═╡ 08e81d69-e179-40b8-a752-e4409dc38f9f
md"##### $\beta$"

# ╔═╡ 8d84a24c-b170-45ee-bc99-8d482d417837
@bind β Slider(0.01:0.01:2.0, default = 1.0)

# ╔═╡ eee2f47b-b50f-4b27-a99f-75b6598f744c
β

# ╔═╡ 02445394-5595-4ecd-97d9-a8c85646a209
md"##### $\gamma$"

# ╔═╡ 836c2113-fe56-4da5-b679-9cbb0e7e77b9
@bind γ Slider(-40:0.1:40, default = 0)

# ╔═╡ 624cd52c-32d5-45c8-b1b7-081861f6fce1
γ

# ╔═╡ 4dbbdbbd-dbba-4c78-baec-a67093d747d1
md"#### Range of $x$"

# ╔═╡ 208ec7c3-bc7e-4436-8296-ebf80dd312a6
x = range(-60, 60, step = 0.02)

# ╔═╡ 22f39d45-8732-45c2-b9f2-3ad1b2cdb4e2
begin
plot(x, [J.(x, α, β, γ), Jhat.(x, α, β, γ)],
	title = "Original quadratic approximation",
	# title = "(α, β, γ) = (1, 1, -6)",
	xlabel = "x",
	ylabel = "Value",
	labels = ["J(x)" "Jhat(x)"],
	xlimits = (-25, 25),
	ylimits = (-200, 100))
plot!([γ], seriestype = :vline, labels = "\\gamma")
end

# ╔═╡ 9cb9264d-303c-4b76-a74b-255e5f6ee608
md"## Negative posterior precision

The main problem with this original approach is that posterior precision $\pi_J$ can algorithmically become negative, which however is logically impossible, and at which point the approximation therefore breaks down. If this happens on at least one occasion in the course of filtering a time series, then the chosen parameter set is out of bounds and another has to be used where posterior precision never turns negative. This places boundaries in parameter space beyond which the algorithm cannot go. While in many cases this is only a minor nuisance, it would still be desirable to have a quadratic approximation that works for all parameter choices in that it never produces negative posterior precision.

To find such an improved approximation, we take a closer look at $J$ and its derivatives. Since posterior precision is the negative of the second derivative, it is especially important to understand how and where this can turn positive. In geometric terms, we need to find the regions of $x$ where the graph of $J$ performs a leftward turn for increasing $x$."

# ╔═╡ 4a03a111-9122-45e3-8a79-3e55773e6829
md"## Variational energy and its first two derivatives"

# ╔═╡ a0434bd9-a497-4b74-a575-b5d09b65b619
md"#### Overview"

# ╔═╡ 8957a5c6-ad3e-43fb-adeb-5a993b034333
plot(x, [J.(x, α, β, γ), dJ.(x, α, β, γ), ddJ.(x, α, β)],
	title = "Variational energy and its first two derivatives",
	# title = "(α, β, γ) = (1, 1, -6)",
	xlabel = "x",
	ylabel = "Value",
	labels = ["J(x)" "J'(x)" "J''(x)"],
	xlimits = (-25, 25),
	ylimits = (-200, 100))

# ╔═╡ a948dccd-1c11-4225-a508-86b17a40c2c2
md"#### First derivative"

# ╔═╡ 057261b0-e21a-45fe-ba61-8aca806db84b
plot(x, dJ.(x, α, β, γ),
	xlabel = "x",
	ylabel = "Value",
	labels = "J'(x)")

# ╔═╡ 65cf3afe-a10c-4fc7-b3a0-3544254e0902
md"#### Second derivative"

# ╔═╡ 353166a1-6a15-4739-b788-27eaf66eec23
md"When $\alpha$ and $\beta$ are such that the variational energy $J$ has a convex region, the second derivate has two extrema which can analytically be calculated, a maximum and a minimum. In terms of negative posterior precision, the problematic region is that around the maximum where the second derivative becomes positive."

# ╔═╡ cef637c1-22f8-4839-9b31-1f709c7ab32a
argmaxddJ(α, β) = log(1/(α + β)*(2*α*β - α*√(α^2 + 3*β^2)))

# ╔═╡ f7fe7e1d-521c-4f18-92f2-eb756d9fb39b
argminddJ(α, β) = log(1/(α + β)*(2*α*β + α*√(α^2 + 3*β^2)))

# ╔═╡ 8adf7697-1397-4a8a-b126-3af3fc37a1c8
begin
plot(x, ddJ.(x, α, β),
	xlabel = "x",
	ylabel = "Value",
	labels = "J''(x)")
plot!([argmaxddJ(α, β)], seriestype = :vline, labels = "maximum")
plot!([argminddJ(α, β)], seriestype = :vline, labels = "minimum")
end

# ╔═╡ cf007872-c859-4969-ba64-c33570bc6645
md"## Modified quadratic approximation

With the original approach, we need to *bound* the allowed parameter region so that it only includes points where the variational energy's second derivative is negative. To avoid this, we need an improved quadratic approximation. We take this in two steps. First, we eliminate the problem of negative posterior precision. Once this is done, we ensure that the quadratic approximation always leads to a Gaussian which closely approximates the non-Gaussian variational posterior."

# ╔═╡ 18cc0ffa-8b66-49c1-aa32-03d2fe473a19
md"### Ensuring positive posterior precision

In order to understand how negative posterior precision can arise in our original approach, we look at the three summands of the variational energy separately:

$$J_1(x) := - \frac{1}{2} \log \left( \alpha + e^x \right)$$"

# ╔═╡ a27115db-8a2f-4a07-bf05-183c2c454b7e
J1(x, α) = - 0.5 * log(α + exp(x))

# ╔═╡ 767335ff-5ee8-4648-80d6-554984c4dd5a
@bind αj1 Slider(0.005:0.005:2.0, default = 1.0)

# ╔═╡ 54594a4f-f631-4327-8a12-336851e73490
plot(x, [J1.(x, αj1)],
	title = "α = 1",
	xlabel = "x",
	ylabel = "Value",
	labels = "J1(x)")

# ╔═╡ a8cba980-ba05-42d3-8fa4-18e6f62fa061
αj1

# ╔═╡ f50c2e8f-d8e4-4944-a7b4-e2ed121bb29d
md"The first summand is concave, approaching the constant $-\frac{1}{2} \log(\alpha)$ for small $x$ and the linear $-\frac{1}{2}x$ for large $x$. Its second derivative is negative everywhere, so it cannot contribute to negative posterior precision.

Next, we have

$$J_2(x) := -\frac{1}{2} \frac{\beta}{\alpha + e^x}.$$"

# ╔═╡ 16799378-67f6-4c59-a1b4-44cdbf2ba088
J2(x, α, β) = -0.5 * β / (α + exp(x))

# ╔═╡ 4317a28a-9090-4be4-9536-19854bf6f971
@bind αj2 Slider(0.005:0.005:2.0, default = 1.0)

# ╔═╡ 25a4a662-9d34-40e7-8a44-cd15c945526c
αj2

# ╔═╡ 896a69fb-9dc5-4418-969b-5a13260a5e92
@bind βj2 Slider(0.01:0.01:2.0, default = 1.0)

# ╔═╡ f498fa3b-c0ba-4ff5-8f3c-3506929426e6
plot(x, J2.(x, αj2, βj2),
	title = "J_2",
	xlabel = "x",
	ylabel = "Value",
	labels = "J2(x)",
    xlimits = (-25, 25))

# ╔═╡ ef21cf16-899b-47c6-a676-0e18f455686e
βj2

# ╔═╡ c3e16dd3-a8a0-40d7-b404-744419426c9d
md"This summand is partly convex and can therefore give rise to regions of negative posterior precision. However, it is clear that this can only happen in a narrowly confined interval of $x$. To the left of this, $J_2$ effectively amounts to the constant $-\beta/2\alpha$, and to the right of this, to $0.$ The transition between these two regimes is centred at roughly $x = \log \alpha$."

# ╔═╡ 5855d910-ecbe-4d76-9438-ae5b042255cc
md"Finally, we have

$$J_3(x) := - \frac{1}{4} (x - \gamma)^2.$$"

# ╔═╡ 25d8f33b-e211-4cd1-a1c0-ffff4f428158
J3(x, γ) = - 0.25 * (x - γ)^2

# ╔═╡ ecafcd3d-ecd8-4401-9b44-2c68b9648fcb
md"This is a quadratic function, which implies that its second derivative is constant. In our case, $J_3''$ is $-1/2$, which means that $J_3$, like $J_1$, cannot give rise to negative posterior precision.

The full picture thus is that $J$ is quadratic both to the left and to the right, and that the only difference between these two regions is the linear term $-\frac{1}{2}x$ contributed by $J_1$ which is present on the right but not the left. Crucially, this term does not affect the second derivative, meaning that posterior precision is the same whether calculated by expanding $J$ to second order on the right or on the left. The only problematic region is in the middle, where $J_2$ makes the transition from $-\beta/2\alpha$ to $0.$"

# ╔═╡ 12cbddbb-634e-4ab6-90cd-8e790e02c404
plot(x, [J.(x, α, β, γ), J1.(x, α), J2.(x, α, β), J3.(x, γ)],
	title = "Variational energy and its three component terms",
	xlabel = "x",
	ylabel = "Value",
	labels = ["J(x)" "J1(x)" "J2(x)" "J3(x)"],
	xlimits = (-25, 25),
	ylimits = (-200, 100))

# ╔═╡ 97cea74b-1121-4388-a7da-16e6a9abf5ae
md"With the problem laid out in this manner, there is now an obvious solution to negative posterior precision as a result of the quadratic approximation: we disregard the offending term $J_2$ when calculating precision. This leaves us with the *concave canonical variational energy* $K$:

$$\begin{align}
K(x) :=& J_1(x) + J_3(x) \\
=& - \frac{1}{2} \log \left( \alpha + e^x \right) - \frac{1}{4} (x - \gamma)^2
\end{align}$$"

# ╔═╡ f127a679-cb79-4115-9015-f60a78030c80
md"The first and second derivatives of $K$ are:

$$\begin{align}
	K'(x) =& - \frac{1}{2} \frac{e^x}{\alpha + e^x} - \frac{1}{2} (x - \gamma)\\[1em] =& - \frac{1}{2} w(x) - \frac{1}{2} (x - \gamma)\\[2em]
	K''(x) =& - \frac{1}{2} \frac{\alpha e^x}{\left( \alpha + e^x \right)^2} - \frac{1}{2}\\[1em] =& - \frac{1}{2} w(x) (1 - w(x)) - \frac{1}{2}
\end{align}$$

Since $\alpha$ is positive, the second derivative is negative everywhere, and negative posterior precision cannot occur. This enables us to construct a quadratic approximation $L_1$ where the precision $\pi_{L_1}$ is $-K''(\gamma)$ and the mean $\mu_{L_1}$ remains as before except that it uses the precision according to its new definition $\pi_{L_1}$

$$L_1(x) = - \frac{\pi_{L_1}}{2} \left(x - \mu_{L_1} \right)^2$$

with

$$\begin{align}
\pi_{L_1} =& - K''(\gamma) = \frac{1}{2} w(\gamma)(1 - w(\gamma)) + \frac{1}{2} \\[2em]
\mu_{L_1} =& \gamma - \frac{J'(\gamma)}{K''(\gamma)} = \gamma + \frac{w(\gamma)}{2 \pi_{L_1}} \delta(γ)
\end{align}$$"


# ╔═╡ bea72981-57cb-406e-ae1c-110b65c7bd74
πL1(α, γ) = 0.5 * w(α, γ) * (1 - w(α, γ)) + 0.5

# ╔═╡ 325dd13d-7262-487e-9cf9-c67279f12f50
μL1(α, β, γ) = γ + 0.5 / πL1(α, γ) * w(α, γ) * δ(α, β, γ)

# ╔═╡ 20f8df66-5e18-4e34-9809-274413e3c758
L1(x, α, β, γ) = -0.5 * πL1(α, γ) * (x - μL1(α, β, γ))^2

# ╔═╡ 40ddc028-6a76-4e9b-81e0-9600e4dacebf
md"### Dealing with the case of two local maxima in the variational energy

For some combinations of $\alpha$, $\beta$, and $\gamma$, the term $J_2$ does not only give rise to a convex region in the variational energy $J$ but also to a second local maximum. In these cases, the variational posterior, obtained by exponentiation and normalization of the variational energy, becomes bimodal. To ensure that our quadratic approximation still produces a Gaussian posterior which reflects the variational posterior as faithfully as possible, we introduce a second quadratic expansion $L_2$ at the approximate location of the second mode.

#### Finding the second mode via the Lambert $W$ function

The stationary points of $J$ satisfy $J'(x) = 0$, i.e.

$$\tfrac{1}{2} w(x) \, \delta(x) = \tfrac{1}{2} (x - \gamma).$$

This transcendental equation has no closed-form solution in general. However, in the limit $\alpha \to 0$ — precisely the regime where the second mode appears and where the original quadratic approximation fails — we have $w(x) \to 1$ and $\delta(x) \to \beta\,e^{-x} - 1$, and the mode equation reduces to

$$\beta \, e^{-x} = 1 + x - \gamma.$$

Substituting $u := 1 + x - \gamma$ gives $u\,e^{u} = \beta\,e^{1 - \gamma}$, which is solved in closed form by the principal branch $W_0$ of the Lambert $W$ function:

$$x^* = \gamma - 1 + W_0\!\left(\beta \, e^{1 - \gamma}\right).$$

Since $\beta \, e^{1-\gamma} > 0$, the solution is unique and real. $W_0$ can be evaluated to machine precision in a few iterations of Halley's method. The approximation is exact as $\alpha \to 0$ and remains accurate whenever $\alpha \ll e^{x^*}$, which is precisely the regime where the bimodal case arises."

# ╔═╡ 81065521-3b4f-473e-804e-15e55520994a
"""Principal branch of the Lambert W function via Halley's method."""
function W0(z::Real)
	z < 0.0   && return NaN
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

# ╔═╡ cb0510f8-9049-4b64-9cd2-be7ff7c8c39a
# Approximate second mode of J, valid for small α.
xstar(β, γ) = γ - 1.0 + W0(β * exp(1.0 - γ))

# ╔═╡ 084a42be-0e72-468a-8262-2ad9743d7aea
md"#### Second quadratic expansion at $x^*$

Because $x^*$ approximates a local maximum of $J$, the second derivative $J''(x^*)$ is negative there, so the quadratic expansion $L_2$ has positive precision:

$$L_2(x) = - \frac{\pi_{L_2}}{2} \left(x - \mu_{L_2} \right)^2$$

with

$$\begin{align}
\pi_{L_2} =& - J''(x^*) = \tfrac{1}{2} w(x^*) \!\left( w(x^*) + (2w(x^*) - 1)\,\delta(x^*) \right) + \tfrac{1}{2} \\[1em]
\mu_{L_2} =& x^* - \frac{J'(x^*)}{J''(x^*)} = x^* + \frac{1}{\pi_{L_2}} \left( \tfrac{1}{2} w(x^*)\,\delta(x^*) - \tfrac{1}{2}(x^* - \gamma) \right).
\end{align}$$

We use the full second derivative $J''$ (not $K''$) at $x^*$ because the curvature is concave there. The residual $J'(x^*)$ — generally small because $x^*$ is close to the true mode — is corrected by the Newton step. In the unlikely case that $\pi_{L_2} \le 0$ (which can occur when $x^*$ falls far from the actual mode), we fall back to $\pi_{L_2} = -K''(x^*)$, which is always positive."

# ╔═╡ 81065521-3b4f-473e-804e-15e55520994b
function πL2(α, β, γ)
	xs = xstar(β, γ)
	π  = - ddJ(xs, α, β)
	return π > 0 ? π : 0.5 * w(α, xs) * (1 - w(α, xs)) + 0.5
end

# ╔═╡ cb0510f8-9049-4b64-9cd2-be7ff7c8c39b
μL2(α, β, γ) = xstar(β, γ) + dJ(xstar(β, γ), α, β, γ) / πL2(α, β, γ)

# ╔═╡ 084a42be-0e72-468a-8262-2ad9743d7aeb
L2(x, α, β, γ) = -0.5 * πL2(α, β, γ) * (x - μL2(α, β, γ))^2

# ╔═╡ ccf4672a-f000-4349-8ec2-aebfa12fbc38
md"#### Softmax blending

We weight the two expansions according to the variational energy $J$ at their respective means:

$$b = \frac{1}{1 + \exp\!\left(J(\mu_{L_1}) - J(\mu_{L_2})\right)}.$$

This assigns higher weight to whichever expansion sits at a point of higher variational energy, i.e. closer to the dominant mode. When the posterior is unimodal near $\gamma$ we have $J(\mu_{L_1}) > J(\mu_{L_2})$ and $b \approx 0$, recovering $L_1$; when the second mode dominates, $J(\mu_{L_2}) > J(\mu_{L_1})$ and $b \approx 1$."

# ╔═╡ 22f0d3d0-d11b-4754-a1e2-af55a963d139
b(α, β, γ) = 1.0 / (1.0 + exp(J(μL1(α, β, γ), α, β, γ) - J(μL2(α, β, γ), α, β, γ)))

# ╔═╡ 6c81f68a-96cd-45a5-981e-3a5765168c61
md"#### Gaussian mixture moment matching

We treat $L_1$ and $L_2$ as the two components of a Gaussian mixture with weights $1-b$ and $b$, and compute the moment-matched mean and variance:

$$\begin{align}
\mu_L       =& (1 - b)\,\mu_{L_1} + b\,\mu_{L_2} \\[1em]
\sigma_L^2  =& \frac{1 - b}{\pi_{L_1}} + \frac{b}{\pi_{L_2}} + b(1 - b)\left(\mu_{L_1} - \mu_{L_2}\right)^2 \\[1em]
\pi_L       =& 1/\sigma_L^2.
\end{align}$$

The between-component term $b(1-b)(\mu_{L_1} - \mu_{L_2})^2$ ensures the resulting Gaussian widens appropriately when both expansions carry substantial weight and their means are far apart. When $b \approx 0$ or $b \approx 1$ this term vanishes and we recover one of the individual expansions. By construction, $\pi_L$ is strictly positive."

# ╔═╡ caf44960-5170-44be-93ce-16dbd748a3bc
μL(α, β, γ) = (1 - b(α, β, γ)) * μL1(α, β, γ) + b(α, β, γ) * μL2(α, β, γ)

# ╔═╡ cd8dd129-d648-4630-92c1-7c261a8dfd04
function πL(α, β, γ)
	bw = b(α, β, γ)
	m1 = μL1(α, β, γ)
	m2 = μL2(α, β, γ)
	σ² = (1 - bw) / πL1(α, γ) + bw / πL2(α, β, γ) + bw * (1 - bw) * (m1 - m2)^2
	return 1 / σ²
end

# ╔═╡ 6b5c4e39-0064-42ea-82d3-230266d11678
md"This gives us our final Gaussian approximation $L$:

$$L(x) = - \frac{\pi_L}{2} \left(x - \mu_L \right)^2.$$

Negative posterior precision is now impossible by construction, and $L$ closely tracks the variational energy $J$ even for extreme combinations of $(\alpha, \beta, \gamma)$ such as arise when a highly confident prediction meets a very large prediction error."

# ╔═╡ 8fd4f3f5-6a95-41d8-8517-b1a7a289d5d5
L(x, α, β, γ) = -0.5 * πL(α, β, γ) * (x - μL(α, β, γ))^2

# ╔═╡ aa612439-4c7c-4da9-a7e2-f8cdd9fef503
md"## Variational posterior and Gaussian approximation

The (unnormalized) marginal posterior distribution under the mean field approximation is the exponential of the variational energy, which we can now compare visually with our Gaussian approximation.

The interactive figures below show the variational energy, $J$, our new approximation $L$, and its components $L_1$ and $L_2$ in logarithmic space, where the approximations are quadratic (above) as well as in native space as probability densities, where the approximations are Gaussian.

It is instructive to start a visual exploration of the approximation's behaviour with $\alpha$ and $\beta$ both set to 1 while varying $\gamma$ from -40 to +40. Throughout this range, $L_1$ offers an excellent approximation which in native probability space is almost indistinguishable from the variational posterior. This reflects a situation where the previous posterior uncertainty $\sigma_\dagger^0$ about the child node (i.e., $\alpha$) is on a similar scale as the total posterior uncertainty $\sigma_\dagger + \left( \mu_\dagger - \mu_\dagger^0 \right)^2$ about that node (i.e., $\beta$).

However, if we reduce $\alpha$ to 0.005 while leaving $\beta$ at 1 and again vary $\gamma$, we see that in such a situation where uncertainty was very low relative to the update resulting from a very large prediction error, the interpolation between $L_1$ and $L_2$ becomes relevant and ensures that our approximation $L$ keeps reflecting the variational posterior across the whole range of $\gamma$."

# ╔═╡ 6396bd6f-a2d4-414e-968d-7d4d2c25d2e8
@bind α2 Slider(0.005:0.005:1, default = 1.0)

# ╔═╡ 537b12a1-7ebe-404e-84f3-c402fc52e4e6
α2

# ╔═╡ 69a2fa83-fd5b-4df3-8afc-223800f1efb1
@bind β2 Slider(0.01:0.01:2.0, default = 1.0)

# ╔═╡ 8e433f84-e081-4a3b-b52a-9250272d9725
β2

# ╔═╡ 670b321a-8a5c-42a4-a4e4-de0cf940c9c6
let γs = range(-40, 40; length = 401)
	plot(γs, [b(α2, β2, g) for g in γs],
		title  = "Softmax blend weight b(γ) at (α, β) = ($(α2), $(β2))",
		xlabel = "γ",
		ylabel = "b",
		labels = "b(α, β, γ)",
		ylimits = (-0.05, 1.05))
end

# ╔═╡ f0d90e85-46ce-4f75-8c55-905faebdcacb
@bind γ2 Slider(-40:0.5:40, default = 0)

# ╔═╡ ec700656-dc49-4b09-b85c-6b744930db67
γ2

# ╔═╡ c55d6ae4-6144-41d2-8c0c-85d57486a785
begin
plot(x, [J.(x, α2, β2, γ2), L1.(x, α2, β2, γ2), L2.(x, α2, β2, γ2), L.(x, α2, β2, γ2)],
	title = "Quadratic apprixmations L1, L2, and L",
	xlabel = "x",
	ylabel = "Value",
	labels = ["J(x)" "L1(x)" "L2(x)" "L(x)"],
	xlimits = (-25 + γ2, 25 + γ2),
	ylimits = (-200, 100))
plot!([γ2], seriestype = :vline, labels = "\\gamma")
plot!([μL1(α2, β2, γ2)], seriestype = :vline, labels = "\\mu_L1")
plot!([μL2(α2, β2, γ2)], seriestype = :vline, labels = "\\mu_L2")
end

# ╔═╡ d097abff-8450-4225-9169-db9b59780d59
pJ(x, α, β, γ) = exp(J(x, α, β, γ)) / quadgk(x -> exp(J(x, α, β, γ)), -Inf, Inf)[1]

# ╔═╡ d0c9def4-aef4-4f26-aaed-5d9ead0019ce
pL1(x, α, β, γ) = sqrt(0.5 * πL1(α, γ) / π) * exp(- 0.5 * πL1(α, γ) * (x - μL1(α, β, γ))^2)

# ╔═╡ 2e523ec3-ef55-424d-9662-96e6eb3967cf
pL2(x, α, β, γ) = sqrt(0.5 * πL2(α, β, γ) / π) * exp(- 0.5 * πL2(α, β, γ) * (x - μL2(α, β, γ))^2)

# ╔═╡ 0c2f70b8-09d1-4589-91db-8ac77d175c38
pL(x, α, β, γ) = sqrt(0.5 * πL(α, β, γ) / π) * exp(- 0.5 * πL(α, β, γ) * (x - μL(α, β, γ))^2)

# ╔═╡ a60eb58b-c5c2-4b63-9ff2-ad6068e7ce67
begin
plot(x, [pJ.(x, α2, β2, γ2), pL1.(x, α2, β2, γ2), pL2.(x, α2, β2, γ2), pL.(x, α2, β2, γ2)],
	title = "Variational and approximate Gaussian posteriors",
	xlabel = "x",
	ylabel = "Density", 
	labels = ["pJ(x)" "pL1(x)" "pL2(x)" "pL(x)"],
	xlimits = (-25 + γ2, 25 + γ2))
end

# ╔═╡ 8f7ce267-86fa-437e-a161-8458724f3887
md"## Full equations

We can now turn back to the fully parameterized form $I$ of the variational energy to give the full equations for HGF volatility prediction error updates (cf. Mathys et al., 2011, Eqs 29--34), where $t$ denotes the time elapsed since the previous update.

### First expansion $L_1$ at the prediction $\hat{\mu}$

$$\begin{align}
\pi_{L_1} =& \hat{\pi} + \frac{\kappa_\dagger^2}{2} w_\dagger (1 - w_\dagger) \\[1em]
\mu_{L_1} =& \hat{\mu} + \frac{\kappa_\dagger w_\dagger}{2 \pi_{L_1}} \delta_\dagger
\end{align}$$

with

$$\begin{align}
\hat{\pi} :=& \frac{1}{\sigma^0 + t\exp(\kappa \mu_\ddagger^0 + \omega)} \\[1em]
w_\dagger :=& \frac{t\exp(\kappa_\dagger \hat{\mu} + \omega_\dagger)}{\sigma_\dagger^0 + t\exp(\kappa_\dagger \hat{\mu} + \omega_\dagger)} \\[1em]
\delta_\dagger :=& \frac{\sigma_\dagger + \left( \mu_\dagger - \hat{\mu}_\dagger \right)^2}{\sigma_\dagger^0 + t\exp \left( \kappa_\dagger \hat{\mu} + \omega_\dagger \right)} - 1.
\end{align}$$

### Second expansion $L_2$ at the Lambert-$W$ mode $x^*$

The expansion point $x^*$ is the approximate mode of the variational energy $I$ in the limit $\sigma_\dagger^0 \to 0$. It is found via the Lambert $W_0$ function by first converting to the canonical variable $y = \log t + \kappa_\dagger x + \omega_\dagger$, in which the prior term of $I$ takes the form $-\tfrac{1}{2} \hat{\pi}_y (y - \gamma_c)^2$ with

$$\begin{align}
\gamma_c :=& \log t + \kappa_\dagger \hat{\mu} + \omega_\dagger \\[1em]
\hat{\pi}_y :=& \frac{\hat{\pi}}{\kappa_\dagger^2}.
\end{align}$$

The mode equation in $y$ then yields

$$y^* = \gamma_c - \frac{1}{2\hat{\pi}_y} + W_0\!\left(\frac{\beta}{2\hat{\pi}_y} \, e^{1/(2\hat{\pi}_y) - \gamma_c}\right),$$

where $\beta = \sigma_\dagger + (\mu_\dagger - \hat{\mu}_\dagger)^2$. Converting back to the native variable:

$$x^* = \frac{y^* - \log t - \omega_\dagger}{\kappa_\dagger}.$$

With $w^*$ and $\delta^*$ being $w$ and $\delta$ evaluated at $x^*$ instead of $\hat{\mu}$,

$$\begin{align}
w^* :=& \frac{t\exp(\kappa_\dagger x^* + \omega_\dagger)}{\sigma_\dagger^0 + t\exp(\kappa_\dagger x^* + \omega_\dagger)} \\[1em]
\delta^* :=& \frac{\sigma_\dagger + (\mu_\dagger - \hat{\mu}_\dagger)^2}{\sigma_\dagger^0 + t\exp(\kappa_\dagger x^* + \omega_\dagger)} - 1,
\end{align}$$

the second expansion reads

$$\begin{align}
\pi_{L_2} =& \hat{\pi} + \frac{\kappa_\dagger^2}{2} w^* \left(w^* + (2 w^* - 1) \, \delta^*\right) \\[1em]
\mu_{L_2} =& x^* + \frac{1}{\pi_{L_2}} \left( \frac{\kappa_\dagger}{2} w^*\,\delta^* - \hat{\pi}(x^* - \hat{\mu}) \right).
\end{align}$$

If $\pi_{L_2} \le 0$ we fall back to the concave curvature $-K''(x^*) = \hat{\pi} + \tfrac{\kappa_\dagger^2}{2} w^*(1 - w^*)$, which is always positive.

### Softmax blending and mixture moment matching

The softmax weight and Gaussian mixture moment matching use the full variational energy $I$ in place of the canonical $J$:

$$b = \frac{1}{1 + \exp\!\left(I(\mu_{L_1}) - I(\mu_{L_2})\right)},$$

$$\begin{align}
\mu_L       =& (1 - b)\,\mu_{L_1} + b\,\mu_{L_2} \\[1em]
\sigma_L^2  =& \frac{1 - b}{\pi_{L_1}} + \frac{b}{\pi_{L_2}} + b(1 - b)\,(\mu_{L_1} - \mu_{L_2})^2 \\[1em]
\pi_L       =& 1/\sigma_L^2.
\end{align}$$

A reference implementation of these equations lives in [`src/updates.jl`](../src/updates.jl) of this package. The simulations in the section below exercise them on a real HGF network."

# ╔═╡ 11111111-0000-0000-0000-000000000001
md"""## Comparative simulations

We reproduce the four simulations from the accompanying paper to verify that the new update equations (i) produce Gaussian posteriors that closely approximate the variational posterior, (ii) yield valid filtering results under standard conditions, (iii) survive situations that crash the original equations, and (iv) extend the usable region of parameter space. All simulations use a two-level continuous HGF via the package's `run_hgf` (for sims 2–4) or the canonical approximations $L$ and $\hat J$ defined above (for sim 1)."""

# ╔═╡ 11111111-0000-0000-0000-000000000002
md"""### Simulation 1 — Approximation quality

KL divergence $D_\mathrm{KL}(p \| q)$ between the normalized variational posterior $p$ (numerical integration) and the Gaussian approximation $q$ at 488 parameter combinations: 8 ratios $\beta/\alpha \in \{1, 2, 5, 10, 20, 50, 100, 200\}$ and 61 values of $\gamma \in [-15, 15]$, with $\alpha = 0.005$."""

# ╔═╡ 11111111-0000-0000-0000-000000000003
sim1_results = let
	α_val  = 0.005
	ratios = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
	γ_vals = collect(range(-15.0, 15.0; length = 61))
	xs     = collect(range(-60.0, 60.0; length = 4001))
	dx     = xs[2] - xs[1]

	function kl(p_vals, μ_q, π_q)
		s = 0.0
		log_norm_q = 0.5 * log(π_q / (2π))
		for (i, x) in enumerate(xs)
			p = p_vals[i]
			if p > 1e-300
				log_q = log_norm_q - 0.5 * π_q * (x - μ_q)^2
				s += p * (log(p) - log_q) * dx
			end
		end
		return max(s, 0.0)
	end

	nr = length(ratios); ng = length(γ_vals)
	kl_c = fill(NaN, nr, ng)
	kl_u = fill(NaN, nr, ng)

	for (i, r) in enumerate(ratios)
		β_val = r * α_val
		for (j, γv) in enumerate(γ_vals)
			Z = sum(exp(J(x, α_val, β_val, γv)) for x in xs) * dx
			p_vals = [exp(J(x, α_val, β_val, γv)) / Z for x in xs]

			# Classic (may produce π_J ≤ 0 → crash)
			π_c = πJ(α_val, β_val, γv)
			if π_c > 0
				kl_c[i, j] = kl(p_vals, μJ(α_val, β_val, γv), π_c)
			end

			# uHGF
			kl_u[i, j] = kl(p_vals, μL(α_val, β_val, γv), πL(α_val, β_val, γv))
		end
	end

	ok_c = .!isnan.(kl_c)
	n_ok = sum(ok_c); n_total = nr * ng
	mean_kl_c = mean(kl_c[ok_c]); mean_kl_u = mean(kl_u)
	summary = @sprintf(
		"Classic succeeded on %d of %d parameter sets (%.1f%%); uHGF succeeded on all %d (100%%). Mean KL divergence where both succeeded: classic = %.4f, uHGF = %.4f.",
		n_ok, n_total, 100n_ok/n_total, n_total, mean_kl_c, mean_kl_u)
	(; ratios, γ_vals, kl_c, kl_u, n_total, n_ok, mean_kl_c, mean_kl_u, summary)
end

# ╔═╡ 11111111-0000-0000-0000-000000000004
md"""**Sim 1 results.** $(sim1_results.summary)"""

# ╔═╡ 11111111-0000-0000-0000-000000000005
let r = sim1_results
	kl_c_plot = clamp.(r.kl_c, 0.0, 5.0)
	kl_c_plot[isnan.(kl_c_plot)] .= 5.0
	kl_u_plot = clamp.(r.kl_u, 0.0, 5.0)
	nr = length(r.ratios)
	p1 = heatmap(r.γ_vals, 1:nr, kl_c_plot;
		xlabel = L"\gamma", ylabel = L"\beta/\alpha",
		yticks = (1:nr, string.(r.ratios)),
		title = "Classic HGF", titlefontsize = 9,
		colorbar_title = L"D_\mathrm{KL}", clims = (0, 5), color = :viridis)
	p2 = heatmap(r.γ_vals, 1:nr, kl_u_plot;
		xlabel = L"\gamma", ylabel = L"\beta/\alpha",
		yticks = (1:nr, string.(r.ratios)),
		title = "uHGF", titlefontsize = 9,
		colorbar_title = L"D_\mathrm{KL}", clims = (0, 5), color = :viridis)
	plot(p1, p2; layout = (1, 2), size = (900, 320),
		plot_title = "Approximation quality", plot_titlefontsize = 10,
		plot_titlefontweight = :bold, margin = 5Plots.mm, bottom_margin = 7Plots.mm)
end

# ╔═╡ 11111111-0000-0000-0000-000000000006
suffstat_data = readdlm(joinpath(@__DIR__, "..", "data", "suff_stat.csv"), ',', Float64)

# ╔═╡ 11111111-0000-0000-0000-000000000007
md"""### Simulation 2 — Filtering under standard conditions

We filter a 320-observation time series (from Mathys, 2020) with $\omega_1 = 2$, $\omega_2 = -1$, and $\alpha_u = 1000$. Both methods are expected to complete successfully."""

# ╔═╡ 11111111-0000-0000-0000-000000000008
sim2_results = let
	u       = suffstat_data[:, 1]
	mu_true = suffstat_data[:, 2]
	sd_true = suffstat_data[:, 3]
	T       = length(u)
	params  = HGFParams(
		[0.0, 1.0], [200.0, 1.0], [0.0, 0.0], [1.0], [2.0], exp(-1.0), 1000.0)
	traj_c = run_hgf(u, params, ClassicUpdate())
	traj_u = run_hgf(u, params, UHGFUpdate())
	first_nan(tr) = something(
		findfirst(i -> any(isnan, tr.mu[i, :]), 1:T), 0)
	crash_c = first_nan(traj_c); crash_u = first_nan(traj_u)
	rmse1 = sqrt(mean((traj_c.mu[:, 1] .- traj_u.mu[:, 1]).^2))
	dmax2 = maximum(abs.(traj_c.mu[:, 2] .- traj_u.mu[:, 2]))
	summary = @sprintf(
		"Classic: %s.  uHGF: %s.  Level-1 RMSE(μ) = %.2f;  Level-2 max|Δμ| = %.2f.",
		crash_c == 0 ? "OK" : "crashed at $crash_c",
		crash_u == 0 ? "OK" : "crashed at $crash_u",
		rmse1, dmax2)
	(; u, mu_true, sd_true, T, traj_c, traj_u, crash_c, crash_u, rmse1, dmax2, summary)
end

# ╔═╡ 11111111-0000-0000-0000-000000000009
md"""**Sim 2 results.** $(sim2_results.summary)"""

# ╔═╡ 11111111-0000-0000-0000-00000000000a
let r = sim2_results
	ks = 1:r.T
	sa_c = 1.0 ./ r.traj_c.pi
	sa_u = 1.0 ./ r.traj_u.pi

	p1 = plot(; xlabel = "Step", ylabel = L"\mu_1",
		title = "Level 1", titlefontsize = 9, legend = :topleft, legendfontsize = 6)
	plot!(p1, ks, r.mu_true; ribbon = 2 .* r.sd_true,
		fillalpha = 0.15, fillcolor = :grey, linecolor = :grey,
		linewidth = 0.5, label = "Ground truth ± 2 SD")
	scatter!(p1, ks, r.u; markersize = 1.5, markercolor = :black,
		markerstrokewidth = 0, alpha = 0.5, label = "Observations")
	plot!(p1, ks, r.traj_c.mu[:, 1]; ribbon = 2 .* sqrt.(sa_c[:, 1]),
		fillalpha = 0.15, fillcolor = :blue, linewidth = 1.5,
		color = :blue, label = "Classic HGF")
	plot!(p1, ks, r.traj_u.mu[:, 1]; ribbon = 2 .* sqrt.(sa_u[:, 1]),
		fillalpha = 0.15, fillcolor = :red, linewidth = 1.5,
		color = :red, label = "uHGF")

	p2 = plot(; xlabel = "Step", ylabel = L"\mu_2",
		title = "Level 2", titlefontsize = 9, legend = :topleft, legendfontsize = 6)
	plot!(p2, ks, r.traj_c.mu[:, 2]; ribbon = 2 .* sqrt.(sa_c[:, 2]),
		fillalpha = 0.15, fillcolor = :blue, linewidth = 1.5,
		color = :blue, label = "Classic HGF")
	plot!(p2, ks, r.traj_u.mu[:, 2]; ribbon = 2 .* sqrt.(sa_u[:, 2]),
		fillalpha = 0.15, fillcolor = :red, linewidth = 1.5,
		color = :red, label = "uHGF")

	plot(p1, p2; layout = (2, 1), size = (700, 500),
		plot_title = "Filtering under moderate volatility (ω₁=2, ω₂=−1, α_u=1000)",
		plot_titlefontsize = 9, plot_titlefontweight = :bold,
		margin = 4Plots.mm, left_margin = 6Plots.mm)
end

# ╔═╡ 11111111-0000-0000-0000-00000000000b
md"""### Simulation 3 — Robustness under extreme prediction errors

The same reference time series, now with high meta-volatility ($\omega_2 = 2$). Large prediction errors at level 1 propagate forcefully to level 2 and are expected to crash the classic HGF."""

# ╔═╡ 11111111-0000-0000-0000-00000000000c
sim3_results = let
	u       = suffstat_data[:, 1]
	mu_true = suffstat_data[:, 2]
	sd_true = suffstat_data[:, 3]
	T       = length(u)
	params  = HGFParams(
		[0.0, 1.0], [200.0, 1.0], [0.0, 0.0], [1.0], [2.0], exp(2.0), 1000.0)
	traj_c = run_hgf(u, params, ClassicUpdate())
	traj_u = run_hgf(u, params, UHGFUpdate())
	first_nan(tr) = something(
		findfirst(i -> any(isnan, tr.mu[i, :]), 1:T), 0)
	crash_c = first_nan(traj_c); crash_u = first_nan(traj_u)
	min_pi2 = crash_u == 0 ? minimum(traj_u.pi[:, 2]) : NaN
	summary = @sprintf(
		"Classic: %s.  uHGF: %s.  Min π₂ (uHGF) = %.4f.",
		crash_c == 0 ? "OK" : "crashed at step $crash_c",
		crash_u == 0 ? "OK — all $T steps" : "crashed at step $crash_u",
		min_pi2)
	(; u, mu_true, sd_true, T, traj_c, traj_u, crash_c, crash_u, min_pi2, summary)
end

# ╔═╡ 11111111-0000-0000-0000-00000000000d
md"""**Sim 3 results.** $(sim3_results.summary)"""

# ╔═╡ 11111111-0000-0000-0000-00000000000e
let r = sim3_results
	ks = 1:r.T
	valid_c = r.crash_c > 0 ? r.crash_c - 1 : r.T

	p1 = plot(; xlabel = "Step", ylabel = L"\mu_1",
		title = "Level 1", titlefontsize = 9, legend = :topleft, legendfontsize = 6)
	plot!(p1, ks, r.mu_true; ribbon = 2 .* r.sd_true,
		fillalpha = 0.15, fillcolor = :grey, linecolor = :grey,
		linewidth = 0.5, label = "Ground truth ± 2 SD")
	scatter!(p1, ks, r.u; markersize = 1.5, markercolor = :black,
		markerstrokewidth = 0, alpha = 0.5, label = "Observations")
	if valid_c > 0
		sa_c = 1.0 ./ r.traj_c.pi[1:valid_c, :]
		plot!(p1, 1:valid_c, r.traj_c.mu[1:valid_c, 1];
			ribbon = 2 .* sqrt.(sa_c[:, 1]),
			fillalpha = 0.15, fillcolor = :blue, linewidth = 1.5,
			color = :blue, label = "Classic HGF")
	end
	if r.crash_c > 0
		vline!(p1, [r.crash_c]; color = :blue, linestyle = :dot,
			linewidth = 1.0, label = "Classic crash")
	end
	sa_u = 1.0 ./ r.traj_u.pi
	plot!(p1, ks, r.traj_u.mu[:, 1]; ribbon = 2 .* sqrt.(sa_u[:, 1]),
		fillalpha = 0.15, fillcolor = :red, linewidth = 1.5,
		color = :red, label = "uHGF")

	p2 = plot(; xlabel = "Step", ylabel = L"\mu_2",
		title = "Level 2", titlefontsize = 9, legend = :topleft, legendfontsize = 6)
	if valid_c > 0
		sa_c = 1.0 ./ r.traj_c.pi[1:valid_c, :]
		plot!(p2, 1:valid_c, r.traj_c.mu[1:valid_c, 2];
			ribbon = 2 .* sqrt.(sa_c[:, 2]),
			fillalpha = 0.15, fillcolor = :blue, linewidth = 1.5,
			color = :blue, label = "Classic HGF")
	end
	if r.crash_c > 0
		vline!(p2, [r.crash_c]; color = :blue, linestyle = :dot,
			linewidth = 1.0, label = "Classic crash")
	end
	plot!(p2, ks, r.traj_u.mu[:, 2]; ribbon = 2 .* sqrt.(sa_u[:, 2]),
		fillalpha = 0.15, fillcolor = :red, linewidth = 1.5,
		color = :red, label = "uHGF")

	plot(p1, p2; layout = (2, 1), size = (700, 500),
		plot_title = "Robustness under extreme prediction errors (ω₁=2, ω₂=2, α_u=1000)",
		plot_titlefontsize = 9, plot_titlefontweight = :bold,
		margin = 4Plots.mm, left_margin = 6Plots.mm)
end

# ╔═╡ 11111111-0000-0000-0000-00000000000f
md"""### Simulation 4 — Parameter-space coverage

A fine 181×181 grid over $\omega_1, \omega_2 \in [-16, 2]$ (step 0.1), filtering the same reference series. *This cell evaluates $32\,761$ parameter combinations twice and takes a few minutes.*"""

# ╔═╡ 11111111-0000-0000-0000-000000000010
sim4_results = let
	u = suffstat_data[:, 1]
	om_grid  = collect(range(-16.0, 2.0; step = 0.1))
	lth_grid = collect(range(-16.0, 2.0; step = 0.1))
	n_om = length(om_grid); n_lth = length(lth_grid); n_total = n_om * n_lth

	status = zeros(Int, n_om, n_lth)   # 0 both, 1 classic-only-fails, 2 both-fail, 3 other
	n_classic_ok = 0; n_uhgf_ok = 0

	first_nan(tr, T) = something(
		findfirst(i -> any(isnan, tr.mu[i, :]), 1:T), 0)

	for (i, om) in enumerate(om_grid)
		for (j, lth) in enumerate(lth_grid)
			params = HGFParams(
				[u[1], 0.0], [100.0, 1.0], [0.0, 0.0], [1.0], [om],
				exp(lth), 100.0)
			traj_c = run_hgf(u, params, ClassicUpdate())
			traj_u = run_hgf(u, params, UHGFUpdate())
			ok_c = first_nan(traj_c, length(u)) == 0
			ok_u = first_nan(traj_u, length(u)) == 0
			ok_c && (n_classic_ok += 1)
			ok_u && (n_uhgf_ok += 1)
			status[i, j] = ok_c && ok_u ? 0 : (!ok_c && ok_u ? 1 : (!ok_c && !ok_u ? 2 : 3))
		end
	end
	n_classic_only_fail = sum(status .== 1)
	summary = @sprintf(
		"Classic OK: %d of %d (%.1f%%).  uHGF OK: %d (%.1f%%).  Classic-only failures: %d.",
		n_classic_ok, n_total, 100n_classic_ok/n_total,
		n_uhgf_ok, 100n_uhgf_ok/n_total, n_classic_only_fail)
	(; om_grid, lth_grid, status, n_total, n_classic_ok, n_uhgf_ok,
	   n_classic_only_fail, summary)
end

# ╔═╡ 11111111-0000-0000-0000-000000000011
md"""**Sim 4 results.** $(sim4_results.summary)"""

# ╔═╡ 11111111-0000-0000-0000-000000000012
let r = sim4_results
	fig = heatmap(r.om_grid, r.lth_grid, Float64.(r.status');
		xlabel = L"\omega_1", ylabel = L"\omega_2",
		color = cgrad([:steelblue, :orange], [0.5]),
		clims = (-0.5, 1.5), colorbar = false,
		size = (500, 420), title = "Parameter-space coverage",
		titlefontsize = 9, titlefontweight = :bold)
	scatter!(fig, [NaN], [NaN]; color = :steelblue, markersize = 6,
		markerstrokewidth = 0, label = "Both succeed")
	scatter!(fig, [NaN], [NaN]; color = :orange, markersize = 6,
		markerstrokewidth = 0, label = "Classic HGF fails")
	plot!(fig; legend = :topright)
end

# ╔═╡ Cell order:
# ╟─0b6a202a-1a1c-11ef-38b9-cba623dfd57d
# ╟─02bb8240-a363-413b-a74f-864837038051
# ╟─e908b646-1c22-49c5-983b-c6c6174babe1
# ╟─2c657235-6d9f-4c71-bf6b-9605d0fd8168
# ╠═c0d483b4-73d5-4904-a340-56c2d467dfba
# ╟─5cccfacd-24a4-4085-80e0-49c06e8aaef8
# ╠═040f42f6-cbd4-4028-a773-50188c4888f0
# ╟─68373d5c-4b35-4050-b3fa-39cb66203ccb
# ╟─2185648d-57b5-4d5f-a1f1-1bb04666b64d
# ╠═7f22a754-d136-4572-b778-108971478179
# ╠═eac83e66-13e0-4405-8bca-b70e19fbf208
# ╟─f42b26c0-9968-4a79-83c1-5327240f9d8b
# ╠═9a70fa82-8b3c-4c3c-9950-57c1a98a19ad
# ╠═89eb7f34-e905-4463-a5b4-25ad70e65648
# ╟─6c3777f6-2ff9-4dcf-8e37-83f18985381f
# ╠═2fad2f32-fa62-459c-98e4-40e9d116ea5c
# ╠═4aca9637-1694-4f5c-bef2-304b8b7b49df
# ╟─cd768b8b-84f6-4cd0-973b-e60ae8e9835a
# ╠═e0a70b3f-edf3-45dc-b672-a067d84895f0
# ╟─22f39d45-8732-45c2-b9f2-3ad1b2cdb4e2
# ╟─92243446-65a4-4787-8a73-f1917083c1be
# ╟─ee860cab-e097-46b2-9bd9-1f7a2de45b36
# ╟─4bc193ac-12eb-4889-ad9d-0bb47ab58e8d
# ╟─fb06391c-c460-4732-9cea-87e22ffdd665
# ╟─08e81d69-e179-40b8-a752-e4409dc38f9f
# ╟─8d84a24c-b170-45ee-bc99-8d482d417837
# ╟─eee2f47b-b50f-4b27-a99f-75b6598f744c
# ╟─02445394-5595-4ecd-97d9-a8c85646a209
# ╟─836c2113-fe56-4da5-b679-9cbb0e7e77b9
# ╠═624cd52c-32d5-45c8-b1b7-081861f6fce1
# ╟─4dbbdbbd-dbba-4c78-baec-a67093d747d1
# ╟─208ec7c3-bc7e-4436-8296-ebf80dd312a6
# ╟─9cb9264d-303c-4b76-a74b-255e5f6ee608
# ╟─4a03a111-9122-45e3-8a79-3e55773e6829
# ╟─a0434bd9-a497-4b74-a575-b5d09b65b619
# ╟─8957a5c6-ad3e-43fb-adeb-5a993b034333
# ╟─a948dccd-1c11-4225-a508-86b17a40c2c2
# ╟─057261b0-e21a-45fe-ba61-8aca806db84b
# ╟─65cf3afe-a10c-4fc7-b3a0-3544254e0902
# ╟─353166a1-6a15-4739-b788-27eaf66eec23
# ╟─8adf7697-1397-4a8a-b126-3af3fc37a1c8
# ╠═cef637c1-22f8-4839-9b31-1f709c7ab32a
# ╠═f7fe7e1d-521c-4f18-92f2-eb756d9fb39b
# ╟─cf007872-c859-4969-ba64-c33570bc6645
# ╟─18cc0ffa-8b66-49c1-aa32-03d2fe473a19
# ╠═a27115db-8a2f-4a07-bf05-183c2c454b7e
# ╠═54594a4f-f631-4327-8a12-336851e73490
# ╟─767335ff-5ee8-4648-80d6-554984c4dd5a
# ╠═a8cba980-ba05-42d3-8fa4-18e6f62fa061
# ╟─f50c2e8f-d8e4-4944-a7b4-e2ed121bb29d
# ╠═16799378-67f6-4c59-a1b4-44cdbf2ba088
# ╠═f498fa3b-c0ba-4ff5-8f3c-3506929426e6
# ╟─4317a28a-9090-4be4-9536-19854bf6f971
# ╟─25a4a662-9d34-40e7-8a44-cd15c945526c
# ╟─896a69fb-9dc5-4418-969b-5a13260a5e92
# ╠═ef21cf16-899b-47c6-a676-0e18f455686e
# ╟─c3e16dd3-a8a0-40d7-b404-744419426c9d
# ╟─5855d910-ecbe-4d76-9438-ae5b042255cc
# ╠═25d8f33b-e211-4cd1-a1c0-ffff4f428158
# ╟─ecafcd3d-ecd8-4401-9b44-2c68b9648fcb
# ╟─12cbddbb-634e-4ab6-90cd-8e790e02c404
# ╟─97cea74b-1121-4388-a7da-16e6a9abf5ae
# ╟─f127a679-cb79-4115-9015-f60a78030c80
# ╠═bea72981-57cb-406e-ae1c-110b65c7bd74
# ╠═325dd13d-7262-487e-9cf9-c67279f12f50
# ╠═20f8df66-5e18-4e34-9809-274413e3c758
# ╟─40ddc028-6a76-4e9b-81e0-9600e4dacebf
# ╠═81065521-3b4f-473e-804e-15e55520994a
# ╠═cb0510f8-9049-4b64-9cd2-be7ff7c8c39a
# ╟─084a42be-0e72-468a-8262-2ad9743d7aea
# ╠═81065521-3b4f-473e-804e-15e55520994b
# ╠═cb0510f8-9049-4b64-9cd2-be7ff7c8c39b
# ╠═084a42be-0e72-468a-8262-2ad9743d7aeb
# ╟─ccf4672a-f000-4349-8ec2-aebfa12fbc38
# ╠═22f0d3d0-d11b-4754-a1e2-af55a963d139
# ╟─6c81f68a-96cd-45a5-981e-3a5765168c61
# ╠═caf44960-5170-44be-93ce-16dbd748a3bc
# ╠═cd8dd129-d648-4630-92c1-7c261a8dfd04
# ╟─6b5c4e39-0064-42ea-82d3-230266d11678
# ╠═8fd4f3f5-6a95-41d8-8517-b1a7a289d5d5
# ╟─aa612439-4c7c-4da9-a7e2-f8cdd9fef503
# ╟─6396bd6f-a2d4-414e-968d-7d4d2c25d2e8
# ╠═537b12a1-7ebe-404e-84f3-c402fc52e4e6
# ╟─69a2fa83-fd5b-4df3-8afc-223800f1efb1
# ╠═8e433f84-e081-4a3b-b52a-9250272d9725
# ╟─670b321a-8a5c-42a4-a4e4-de0cf940c9c6
# ╟─f0d90e85-46ce-4f75-8c55-905faebdcacb
# ╠═ec700656-dc49-4b09-b85c-6b744930db67
# ╟─c55d6ae4-6144-41d2-8c0c-85d57486a785
# ╟─a60eb58b-c5c2-4b63-9ff2-ad6068e7ce67
# ╠═d097abff-8450-4225-9169-db9b59780d59
# ╠═d0c9def4-aef4-4f26-aaed-5d9ead0019ce
# ╠═2e523ec3-ef55-424d-9662-96e6eb3967cf
# ╠═0c2f70b8-09d1-4589-91db-8ac77d175c38
# ╟─8f7ce267-86fa-437e-a161-8458724f3887
# ╟─11111111-0000-0000-0000-000000000001
# ╟─11111111-0000-0000-0000-000000000002
# ╠═11111111-0000-0000-0000-000000000003
# ╟─11111111-0000-0000-0000-000000000004
# ╟─11111111-0000-0000-0000-000000000005
# ╠═11111111-0000-0000-0000-000000000006
# ╟─11111111-0000-0000-0000-000000000007
# ╟─11111111-0000-0000-0000-000000000008
# ╟─11111111-0000-0000-0000-000000000009
# ╟─11111111-0000-0000-0000-00000000000a
# ╟─11111111-0000-0000-0000-00000000000b
# ╟─11111111-0000-0000-0000-00000000000c
# ╟─11111111-0000-0000-0000-00000000000d
# ╟─11111111-0000-0000-0000-00000000000e
# ╟─11111111-0000-0000-0000-00000000000f
# ╟─11111111-0000-0000-0000-000000000010
# ╟─11111111-0000-0000-0000-000000000011
# ╟─11111111-0000-0000-0000-000000000012
# ╟─d0173c55-4f11-4856-8756-d9dd9db8ca3b
