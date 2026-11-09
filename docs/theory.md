# The equations as implemented

Notation follows Sandhu et al., *J. Comput. Phys.* 426 (2021) 109728, referred to as
"the paper". Sections 1–3 restate the paper in the form the code uses; sections 4–9
cover what the code adds.

## 1. Setting

The parameters are split as $\phi = \{\phi_\alpha, \phi_{-\alpha}\}$:

- $\phi_\alpha \in \mathbb{R}^{n}$, the *questionable* parameters, carry ARD priors
  $\mathcal{N}(\phi_\alpha \mid 0, A^{-1})$, $A = \mathrm{diag}(\alpha)$;
- $\phi_{-\alpha}$, the *a priori relevant* ones, carry a known prior $p(\phi_{-\alpha})$.

The likelihood is a black box. Its product with the known prior is approximated once
by a Gaussian mixture,

$$p(\mathcal{D}\mid\phi)\,p(\phi_{-\alpha}) \;\approx\; \sum_{k=1}^{K} a_k\, \mathcal{N}(\phi \mid \mu_k, \Sigma_k), \tag{1}$$

built from samples of the left-hand side (`NSBL.sample`, `NSBL.fit_mixture`). Only the
marginal of (1) over $\phi_\alpha$, with means $\mu_{\alpha,k}$ and covariances
$\Sigma_{\alpha,k}$, enters the evidence.

## 2. Evidence, gradient, Hessian

With $B_k = \Sigma_{\alpha,k} + A^{-1}$ and $\theta = \log\alpha$,

$$\hat p(\mathcal{D}\mid\alpha) = \sum_k a_k\, \mathcal{N}(\mu_{\alpha,k}\mid 0, B_k), \qquad
w_k = \frac{a_k\,\mathcal{N}(\mu_{\alpha,k}\mid 0, B_k)}{\hat p(\mathcal{D}\mid\alpha)} .$$

The objective is $\mathcal{L}(\theta) = \log\hat p(\mathcal{D}\mid\theta) + \sum_i (r_i\theta_i - s_i e^{\theta_i})$,
the second term being the Gamma hyperprior. With $v_{k,i} = \partial\log\mathcal{N}(\mu_{\alpha,k}\mid0,B_k)/\partial\theta_i$
and $\bar v = \sum_k w_k v_k$,

$$\frac{\partial\mathcal{L}}{\partial\theta_i} = \bar v_i + r_i - s_i\alpha_i, \qquad
v_{k,i} = \tfrac12\big(\gamma_{k,i} - \alpha_i m_{k,i}^2\big), \qquad \gamma_{k,i} = 1 - \alpha_i P_{k,ii},$$

$$\frac{\partial^2\mathcal{L}}{\partial\theta_i\partial\theta_j} = \sum_k w_k\Big[\alpha_i\alpha_j\big(\tfrac12 P_{k,ij}^2 + m_{k,i}m_{k,j}P_{k,ij}\big) + v_{k,i}v_{k,j}\Big] - \bar v_i\bar v_j + \delta_{ij}\big(\bar v_i - \tfrac12 - s_i\alpha_i\big),$$

where $P_k = (\Sigma_{\alpha,k}^{-1} + A)^{-1}$ and $m_k = P_k\Sigma_{\alpha,k}^{-1}\mu_{\alpha,k}$
are the posterior moments of kernel $k$. These are Eqs. (9), (18), (19) and Table 2 of
the paper. `tests/test_evidence.py` checks them against finite differences and against
a brute-force product of Gaussians.

### Stable form

Evaluated as written, $P_k$ and $m_k$ lose all their digits when a parameter is
pruned: $\alpha_i\to\infty$ gives $P_{k,ii}\to 1/\alpha_i$ and $\gamma_{k,i}$ becomes a
difference of two numbers equal to one. The code never forms $P_k$ from
$\Sigma_{\alpha,k}^{-1}$. Everything is expressed through $B_k^{-1}$, obtained from one
Cholesky factorisation, using

$$\gamma_{k,i} = \frac{(B_k^{-1})_{ii}}{\alpha_i}, \qquad \alpha\odot m_k = B_k^{-1}\mu_{\alpha,k} =: u_k, \qquad A P_k A = A - B_k^{-1} =: Q_k .$$

They follow from $B_k^{-1} = A - A P_k A$ and $m_k = A^{-1}B_k^{-1}\mu_{\alpha,k}$. Then
$v_{k,i} = \tfrac12(\gamma_{k,i} - u_{k,i}^2/\alpha_i)$ and the first Hessian term is
$P_{k,ij}\,(\tfrac12 Q_{k,ij} + u_{k,i}u_{k,j})$ with $P_{k,ij} = Q_{k,ij}/(\alpha_i\alpha_j)$.
No inverse of $\Sigma_{\alpha,k}$ is needed, and $B_k$ is positive definite for every
finite $\alpha$.

### Cost

Kernels that share a covariance share $B_k^{-1}$. For a kernel density estimate all
$K$ kernels share one, so an evaluation costs $O(n^3 + K n^2)$: about 1 ms for
$K = 5000$, $n = 21$. In general the cost is $O(G n^3 + K n^2)$ with $G$ the number of
distinct covariances.

## 3. Relevance and posterior

$\gamma_{k,i}\in[0,1]$ is the share of the posterior precision of $\phi_i$ that comes
from the data in kernel $k$. The kernels are combined by a root mean square weighted
by the posterior weights $w_k$ (`weighting="posterior"`), or uniformly as in Eq. (23)
of the paper (`weighting="uniform"`); the two coincide when the kernels share one
covariance.

The posterior is the mixture $\sum_k w_k\,\mathcal{N}(\phi\mid m_k, P_k)$ over all
parameters, each kernel being updated by a Kalman step with observation operator
$\phi\mapsto\phi_\alpha$ (`GaussianMixture.multiply_gaussian`). With $C_k$ the
cross-covariance between $\phi_\alpha$ and $\phi_{-\alpha}$ in kernel $k$,

$$m_{-\alpha,k} = \mu_{-\alpha,k} - C_k^T B_k^{-1}\mu_{\alpha,k}, \qquad
P_{-\alpha,k} = \Sigma_{-\alpha,k} - C_k^T B_k^{-1} C_k, \qquad
D_k = C_k - \Sigma_{\alpha,k} B_k^{-1} C_k .$$

The last expression differs from Eq. (B.14) of the paper, which carries products of
the kernel means; a posterior covariance of a linear-Gaussian update cannot depend on
the means, and the brute-force check in the test-suite confirms the form above.

## 4. Sign-constrained parameters

A coefficient restricted to $\phi_i \ge 0$ cannot carry a Gaussian ARD prior. It
carries the half-normal one, $2\,\mathcal{N}(\phi_i\mid 0,\alpha_i^{-1})$ on
$\phi_i\ge0$.

Let $g$ be the mixture (1), fitted to samples that all satisfy the constraint, and let
$g_f(\phi) = \sum_{s} g(D_s\phi)$ be its *folded* version on the admissible orthant,
the sum running over the sign flips $D_s$ of the constrained coordinates. $g_f$ is the
classical reflection estimate of a density with a boundary: the mass a kernel leaks
across zero is mirrored back. Since the prior is even in the constrained coordinates,

$$\int_{\phi_c\ge0} g_f(\phi)\,\prod_{i\in c} 2\,\mathcal{N}(\phi_i\mid0,\alpha_i^{-1})\prod_{i\notin c}\mathcal{N}(\phi_i\mid0,\alpha_i^{-1})\,d\phi
\;=\; 2^{n_c}\int_{\mathbb{R}^n} g(\phi)\,\mathcal{N}(\phi_\alpha\mid0,A^{-1})\,d\phi .$$

The right-hand side is the standard formula. So for a sign constraint at zero the
evidence, its gradient and its Hessian are unchanged up to the constant $2^{n_c}$, and
the boundary is treated exactly. The posterior is the folded mixture: draws are taken
from the Gaussian mixture and their constrained coordinates replaced by their absolute
value (`NSBLResult.sample`).

This holds for any mixture, but the folded mixture must be a good estimate of the
constrained density. A kernel density estimate is, by construction. A mixture with few
kernels fitted to samples piled up against zero is not: one Gaussian fitted to
half-normal samples has its mean at $0.8\sigma$ and reports a relevance of about 0.4
for a parameter that should be pruned. For the EM and single-Gaussian mixtures the
samples of such parameters are therefore mirrored about zero before the fit
(`symmetric_dims`), which leaves the evidence integral unchanged and makes the density
smooth across zero.

## 5. Bounded parameters

With `truncate_prior=True` the ARD prior of a parameter confined to $[l_i, u_i]\ni0$ is
the Gaussian truncated to that interval and renormalised. The objective gains the
separable term

$$-\sum_i \log Z_i(\theta_i), \qquad Z_i = \Phi(u_i\sqrt{\alpha_i}) - \Phi(l_i\sqrt{\alpha_i}),$$

whose derivatives, with $t = e^{\theta/2}$ and $\varphi$ the standard normal density,
are

$$\frac{dZ}{d\theta} = \frac t2\big[u\varphi(ut) - l\varphi(lt)\big], \qquad
\frac{d^2Z}{d\theta^2} = \frac t4\big[u\varphi(ut)(1-u^2t^2) - l\varphi(lt)(1-l^2t^2)\big].$$

Without the term, a wide prior is charged for the mass it puts outside the box; with
it, $\alpha_i\to0$ gives the uniform prior on the box and the evidence tends to that of
the uniform-prior model, which is the quantity the sampling stage estimates. For
$l_i = 0$, $Z_i\to\tfrac12$ as $\alpha_i\to\infty$: the factor two of the half-normal
prior. Kernel mass leaking across a *non-zero* bound is not corrected; it is an
approximation error of the mixture there.

## 6. What the bandwidth of a kernel density estimate does

For a kernel density estimate with kernel covariance $h^2 S$ the expected evidence is

$$\mathbb{E}\big[\hat p(\mathcal{D}\mid\alpha)\big] = \int p(\mathcal{D}\mid\phi)\,p(\phi_{-\alpha})\;\mathcal{N}\big(\phi_\alpha \mid 0,\; A^{-1} + h^2 S_\alpha\big)\,d\phi :$$

the estimate is unbiased for a prior of covariance $A^{-1} + h^2 S_\alpha$. Smoothing
acts as a floor on the prior covariance. Two consequences:

- a "pruned" parameter is tested against a narrow Gaussian of width $h\sigma_i$, not
  against zero exactly, which shifts the pruning threshold of a Gaussian posterior
  from $|\mu_i|/\sigma_i < 1$ to $|\mu_i|/\sigma_i < \sqrt{1+h^2}$;
- once $\alpha_i^{-1} \ll h^2 S_{ii}$ the evidence no longer depends on $\alpha_i$. On
  that plateau the relevance indicator runs from one to zero while the evidence is
  flat, so where a local search stops is decided by sampling noise.

A smaller $h$ reduces the bias but the evidence of a sparse model then rests on the
few kernels close to the corresponding subspace: with $n_p$ pruned parameters, roughly
a fraction $h^{n_p}$ of them. In 21 dimensions Scott's rule gives $h\approx0.7$, and
with thirteen pruned parameters this fraction is about one percent. This is why
`fit_mixture` switches to an EM mixture beyond eight parameters.

## 7. Decision rule

Two rules are applied after each Newton run (`maximize_evidence`).

*Minimum evidence gain.* The fitted precision of a term always does at least as well
as pruning it, so a small gain is no evidence that the term is needed; in a linear
model, type-II maximum likelihood keeps any coefficient whose estimate exceeds its own
standard error. Precisions are sent to the pruned state, least relevant first, for as
long as the objective drops by less than `evidence_tol` in total; the others are then
re-optimised. The default is one nat (a Bayes factor of $e$); zero gives the rule of
the paper. This also removes the indeterminacy on the plateau of section 6.

*Scale invariance.* With `standardize=True` the admissible interval of $\theta_i$, the
starting points and the hyperprior are expressed for the dimensionless precision
$\alpha_i\,\mathrm{Var}(\phi_i)$, the variance being that of the mixture. A Gamma
hyperprior with a fixed rate caps the precision a pruned parameter can reach, at a
value expressed in the units of the parameter; a coefficient that the data pin at
$0.02\pm0.01$ is then reported as relevant because the cap is reached before the
prior becomes narrower than the kernels. The default hyperprior is flat in
$\log\alpha$ ($r = s = 0$), bounded by a quadratic barrier at $\pm30$ around the scaled
origin.

`NSBLResult.evidence_gain` reports, for each term, the log-evidence lost when that
term alone is pruned: the log Bayes factor in its favour, given the others.

## 8. Sampler

`SMCSampler` is a tempered sequential Monte Carlo sampler (TMCMC): the likelihood is
raised to a power $\beta$ going from 0 to 1, each step being chosen so that the
effective sample size of the importance weights drops to half the population, and the
population is resampled and moved by Metropolis kernels. The evidence is the product
of the mean importance weights. Three points matter for accuracy.

*Cross-fitting.* The proposals are fitted to the population. Fitted to the particles
they then move, they bias the evidence upwards by $O(d^2/N)$: a factor 1.85 for
$d = 21$, $N = 1000$ on a Gaussian target. The population is split in two halves, each
resampled on its own and moved with the proposal fitted on the other, which removes
the bias (error of the log-evidence $+0.005\pm0.085$ on the same target). The two
halves give two evidence estimates; half their gap is returned as
`log_evidence_error`.

*Reflection at the bounds.* A random-walk step leaving the support is reflected, which
is equivalent to sampling the mirror-symmetric extension of the target. With a
correlated proposal covariance $LL^T$ the reflected step is not the mirror image of the
reverse step, and the Metropolis ratio needs the factor

$$\log\frac{q(y\to x)}{q(x\to y)} = \tfrac12\lVert z\rVert^2 - \tfrac12\lVert L^{-1}(F\odot Lz)\rVert^2,$$

where $z$ is the standard normal draw and $F_j = -1$ for the coordinates reflected an
odd number of times. Without it the evidence of a 21-dimensional target with eight
coefficients piled up against zero was off by $-0.27$ nats; with it, by $+0.007\pm0.03$.

*Independence moves.* When the tempered target is close to Gaussian, proposals drawn
from the Gaussian fitted on the other half refresh a particle completely when
accepted. They are used while their acceptance exceeds 8 %, and every stage ends with
random-walk steps. On a 21-dimensional Gaussian target this divides the number of
likelihood evaluations by 6.7. A stage ends when no coordinate of the population is
correlated by more than 0.25 with its value at the start of the stage, or after
`max_steps` steps; multimodal targets always reach the cap.

## 9. Likelihood of field data

*Reduced misfit.* For a POD emulator $Q = Q_0 + \Phi a(\Gamma)$ and weights $W$ on the
selected points, with the thin QR factorisation $W^{1/2}\Phi = \mathsf{Q}R$ and
$t = \mathsf{Q}^T W^{1/2}(y - Q_0)$,

$$\mathrm{SSE}(\Gamma) = \lVert y - Q_0 - \Phi a(\Gamma)\rVert_W^2 = \lVert R\,a(\Gamma) - t\rVert^2 + \mathrm{SSE}_{\min}.$$

Evaluating the likelihood costs $O(r^2)$ per parameter vector, $r$ being the number of
modes, plus the Kriging prediction of the $r$ coefficients; the field is not
reconstructed. The form is a sum of squares, so it does not lose digits when the fit
is good.

*Error variances.* With an inverse-Gamma prior on the error variance of field $t$
integrated out,

$$\log L(\Gamma) = -\sum_t \Big(a_0 + \frac{n_t}{2}\Big)\log\Big(b_0 + \frac{w_t}{2}\mathrm{SSE}_t(\Gamma)\Big) + \mathrm{const},$$

which for $a_0 = b_0 = 0$ is invariant to the scale of each field: quantities of
different magnitude need no normalisation. With a fixed variance
$\sigma_t^2 = \kappa\,\mathrm{MSE}_t^{\rm baseline}$ instead, $-\log L$ is the
baseline-normalised loss of the deterministic framework times a constant.

*Spatial correlation.* $n_t$ is the effective number of independent observations in
field $t$ and $w_t = n_t/N_t$ the corresponding tempering exponent, $N_t$ being the
number of mesh points. For a correlated Gaussian field the sum of squares has the
variance of a chi-square variable with

$$n_{\rm eff} = \frac{N}{\prod_{\rm axes}\big(1 + 2\sum_{k\ge1}\rho_k^2\big)}$$

degrees of freedom, $\rho_k$ being the autocorrelation at lag $k$ along an axis
(`effective_sample_size`). The autocorrelation is taken about zero, so a uniform bias
counts as one piece of information. $n_{\rm eff}$ is a modelling choice with a strong
effect: it sets the width of the posterior, hence how many terms are found relevant.
Estimating it from the residual of the baseline model is conservative.

## 10. Linear models

For $d = \Psi\phi + \epsilon$, $\epsilon\sim\mathcal{N}(0,\rho^{-1}I)$, with ARD priors on
every coefficient, (1) holds exactly with one kernel: $\mu = (\Psi^T\Psi)^{-1}\Psi^Td$,
$\Sigma = (\rho\Psi^T\Psi)^{-1}$. The evidence is then
$\mathcal{N}(d\mid0,\Psi A^{-1}\Psi^T + \rho^{-1}I)$ and the stationarity condition of
the objective is the SBL update $\alpha_i = (\gamma_i + 2r_i)/(m_i^2 + 2s_i)$.
`tests/test_linear.py` checks both. This requires at least as many data as
coefficients; an under-determined problem has no normalisable likelihood in $\phi$ and
is outside the scope of NSBL.
