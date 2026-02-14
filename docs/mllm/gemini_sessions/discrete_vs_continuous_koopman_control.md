# Discrete vs Continuous Koopman Control: Theoretical Analysis

## Executive Summary

This document analyzes the theoretical foundations of **discrete-time EDMD + DARE** versus **continuous-time kGEDMD + CARE** for Koopman-based control, providing rigorous justification for when each approach is appropriate.

---

## 1. The Core Question

**Observation**: In our experiments:
- **DARE (Discrete ARE)** with polynomial features worked reliably on the double well
- **CARE (Continuous ARE)** often failed or required careful tuning
- The learned eigenvalues didn't match physical Jacobian eigenvalues, yet control still succeeded

**Questions to Address**:
1. Can we theoretically justify the discrete + DARE approach?
2. What are the trade-offs between continuous and discrete formulations?
3. Should we abandon kGEDMD + CARE in favor of discrete EDMD + DARE?

---

## 2. Mathematical Preliminaries

### Definition 2.1 (Koopman Operator)
For a dynamical system $\dot{x} = f(x)$ with flow $\Phi_t$, the **Koopman operator** $\mathcal{K}_t$ acts on observables $g: \mathcal{X} \to \mathbb{R}$ by:
$$(\mathcal{K}_t g)(x) = g(\Phi_t(x))$$

### Definition 2.2 (Koopman Generator)
The **infinitesimal generator** $\mathcal{L}$ of the Koopman semigroup is:
$$\mathcal{L}g = \lim_{t \to 0^+} \frac{\mathcal{K}_t g - g}{t} = f(x) \cdot \nabla g(x)$$

For stochastic systems $dx = b(x)dt + \sigma(x)dW_t$, the generator includes the Itô correction:
$$\mathcal{L}g = b \cdot \nabla g + \frac{1}{2}\text{tr}(\sigma\sigma^\top \nabla^2 g)$$

### Definition 2.3 (Lifted State Space)
Given a dictionary of observables $\psi = [\psi_1, \ldots, \psi_n]^\top: \mathcal{X} \to \mathbb{R}^n$, the **lifted state** is:
$$z = \psi(x) \in \mathbb{R}^n$$

### Definition 2.4 (Discrete vs Continuous Representations)
- **Discrete**: $z_{k+1} = A_d z_k + B_d u_k$ where $A_d = e^{L \Delta t}$
- **Continuous**: $\dot{z} = L z + B u$ where $L \approx \mathcal{L}|_{\text{span}(\psi)}$

---

## 3. Theoretical Foundations

### 3.1 Discrete-Time Formulation (EDMD + DARE)

**Data Model**:
$$z_{k+1} = A_d z_k + B_d u_k + w_k$$

where $(A_d, B_d)$ are discrete-time matrices and $w_k$ is model error.

### Proposition 3.1 (MLE Characterization)
*Least-squares EDMD is the maximum likelihood estimator under Gaussian transition noise.*

**Proof.** Assume $w_k \sim \mathcal{N}(0, \Sigma)$ i.i.d. The likelihood of data $\mathcal{D} = \{(z_k, u_k, z_{k+1})\}_{k=1}^N$ is:
$$p(\mathcal{D} | A, B) = \prod_{k=1}^N \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(z_{k+1} - Az_k - Bu_k)^\top \Sigma^{-1} (z_{k+1} - Az_k - Bu_k)\right)$$

Taking the log and noting that maximizing over $(A, B)$ requires minimizing:
$$\sum_{k=1}^N (z_{k+1} - Az_k - Bu_k)^\top \Sigma^{-1} (z_{k+1} - Az_k - Bu_k)$$

For isotropic $\Sigma = \sigma^2 I$, this reduces to least-squares:
$$\min_{A,B} \sum_{k=1}^N \|z_{k+1} - Az_k - Bu_k\|^2 \quad \blacksquare$$

### Corollary 3.2 (MAP = Ridge Regression)
*With Gaussian prior $\text{vec}(A, B) \sim \mathcal{N}(0, \lambda^{-1} I)$, MAP estimation yields Ridge regression.*

### 3.2 Continuous-Time Formulation (kGEDMD + CARE)

**Generator Estimation (Galerkin):**
$$G_{10} = L \cdot G_{00}$$

where $G_{00}[i,j] = \langle \psi_i, \psi_j \rangle_\mu$ and $G_{10}[i,j] = \langle \mathcal{L}\psi_i, \psi_j \rangle_\mu$.

### Remark 3.3 (Stochastic Systems)
For SDEs, the generator includes:
$$G_{10}[i,j] = \langle b \cdot \nabla\psi_i + \frac{1}{2}\text{tr}(a \nabla^2\psi_i), \psi_j \rangle_\mu$$
where $a = \sigma\sigma^\top$. This requires knowing or estimating the diffusion coefficient.

---

## 4. Spectral Analysis and Control Frequency

This section establishes the fundamental relationship between generator eigenvalues and minimum viable control frequencies.

### Definition 4.1 (Generator Eigenvalue Decomposition)
Let $L \in \mathbb{R}^{n \times n}$ be the finite-dimensional approximation of $\mathcal{L}$. The eigendecomposition is:
$$L = V \Lambda V^{-1}$$
where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ with $\lambda_i \in \mathbb{C}$.

### Definition 4.2 (Characteristic Time Scale)
For eigenvalue $\lambda_i$ with $\text{Re}(\lambda_i) \neq 0$, the **characteristic time scale** is:
$$\tau_i = \frac{1}{|\text{Re}(\lambda_i)|}$$

### Proposition 4.3 (Discrete-Continuous Eigenvalue Correspondence)
*If $\lambda_c$ is an eigenvalue of the continuous generator $L$, then $\lambda_d = e^{\lambda_c \Delta t}$ is the corresponding eigenvalue of the discrete operator $A_d = e^{L \Delta t}$.*

**Proof.** By the spectral mapping theorem for matrix exponentials:
$$\sigma(e^{L \Delta t}) = e^{\sigma(L) \Delta t}$$
where $\sigma(\cdot)$ denotes the spectrum. $\blacksquare$

### Corollary 4.4 (Stability Correspondence)
- $\text{Re}(\lambda_c) < 0 \Leftrightarrow |\lambda_d| < 1$ (stable mode)
- $\text{Re}(\lambda_c) > 0 \Leftrightarrow |\lambda_d| > 1$ (unstable mode)
- $\text{Re}(\lambda_c) = 0 \Leftrightarrow |\lambda_d| = 1$ (marginally stable)

### Theorem 4.5 (Minimum Control Frequency for Unstable Modes)
*Let $\lambda > 0$ be an unstable eigenvalue of the continuous generator $L$. For discrete-time control at frequency $f = 1/\Delta t$ to stabilize the corresponding mode, it is necessary that:*
$$f > \frac{\lambda}{2\ln 2}$$

**Proof.** An unstable mode with eigenvalue $\lambda > 0$ grows as $e^{\lambda t}$. The doubling time is:
$$T_{double} = \frac{\ln 2}{\lambda}$$

For the controller to observe and correct the state before it doubles, we need at least two samples per doubling period (Nyquist-like argument):
$$\Delta t < \frac{T_{double}}{2} = \frac{\ln 2}{2\lambda}$$

Therefore:
$$f = \frac{1}{\Delta t} > \frac{2\lambda}{\ln 2} \approx 2.88 \lambda$$

For practical robustness (allowing control margin), a factor of 2 is typical:
$$f_{practical} > \frac{\lambda}{\ln 2} \approx 1.44\lambda \quad \blacksquare$$

### Remark 4.6 (Control Margin)
The factor of 2 in the Nyquist argument is a minimum. In practice, control systems require margin for:
- Measurement noise
- Model uncertainty  
- Actuator delays

A practical rule is $f \geq 5\lambda$ for robust stabilization.

### Example 4.7 (Double Well System)
For the double well with potential $V(x_1) = (x_1^2 - 1)^2$:
- At origin: Jacobian eigenvalue $\lambda = +4$ (unstable)
- Time scale: $\tau = 0.25$s
- Minimum frequency: $f_{min} > 5.8$Hz
- Practical: $f \geq 20$Hz recommended

### Proposition 4.8 (Spectral Control Design from Generator)
*Given the continuous generator $L$ with spectrum $\{\lambda_i\}$, the control matrices $(Q, R)$ for CARE should satisfy:*
$$\frac{q_{ii}}{r_{jj}} \geq \max_i |\text{Re}(\lambda_i)|^2$$
*to ensure closed-loop stability.*

**Proof Sketch.** The minimum gain to stabilize mode $\lambda_i$ scales with $|\lambda_i|$. The LQR gain $K$ scales as $\sqrt{Q/R}$. Therefore $Q/R \propto \lambda^2$ provides sufficient authority. $\blacksquare$

---

## 5. Coarse-Graining Robustness

### Definition 5.1 (Coarse-Graining: Abstract)
Let $(\mathcal{X}, \mathcal{F}, \mu)$ be a probability space and $\pi: \mathcal{X} \to \mathcal{Y}$ a measurable map onto a "coarser" space $\mathcal{Y}$. **Coarse-graining** is the process of approximating the dynamics on $\mathcal{X}$ by dynamics on $\mathcal{Y}$ via the pushforward:
$$P_Y(A) = P_X(\pi^{-1}(A))$$

Common instantiations:
- **State-space coarse-graining**: $\mathcal{Y}$ is a partition of $\mathcal{X}$ (Ulam's method, set-oriented methods)
- **Observable coarse-graining**: $\mathcal{Y} = \text{span}(\psi_1, \ldots, \psi_n) \subset L^2(\mu)$ (Koopman/EDMD)
- **Temporal coarse-graining**: Dynamics at resolution $\Delta t_1$ applied at resolution $\Delta t_2$

### Remark 5.2 (Connection to Koopman)
Koopman-based methods are a form of **observable coarse-graining**: we project infinite-dimensional dynamics onto a finite dictionary span. The quality of this projection determines control performance.

### Definition 5.3 (Lumpability)
A Markov chain with transition matrix $P$ is **lumpable** with respect to partition $\pi$ if the coarse-grained chain $P_Y$ is also Markov, i.e., transition probabilities depend only on the equivalence class, not the specific state within it.

### Remark 5.4 (Koopman as Exact Lumpability)
If observables $\psi$ span a Koopman-invariant subspace, they define an **exactly lumpable** system: dynamics in $z = \psi(x)$ are self-consistent. EDMD approximates this when exact invariance is unavailable.

### Definition 5.5 (Temporal Coarse-Graining)
**Temporal coarse-graining** is using dynamics learned at resolution $\Delta t_1$ for control at resolution $\Delta t_2 \neq \Delta t_1$.

### Theorem 5.6 (Continuous Model = Temporally Scale-Free)
*If $L$ is the true generator, then control at any frequency $f > f_{min}$ can be synthesized from $L$ via:*
$$A_{\Delta t} = e^{L \Delta t}, \quad B_{\Delta t} = \int_0^{\Delta t} e^{L s} B_c \, ds$$

**Proof.** This is the standard discretization of continuous LTI systems. The exponential is computed exactly (to numerical precision) without approximation error. $\blacksquare$

### Proposition 5.7 (Discrete Model Temporal Scaling Error)
*If $A_d$ is learned at sampling period $\Delta t_1$, using linear rescaling for period $\Delta t_2$:*
$$\tilde{A}_{\Delta t_2} = I + (A_{\Delta t_1} - I)\frac{\Delta t_2}{\Delta t_1}$$
*introduces error $O(\Delta t_2^2)$ in each matrix element.*

**Proof.** True scaling: $A_{\Delta t_2} = e^{L \Delta t_2}$. The approximation uses:
$$\tilde{A} = I + L\Delta t_2 \cdot \frac{A_{\Delta t_1} - I}{L \Delta t_1} = I + (e^{L\Delta t_1} - I)\frac{\Delta t_2}{\Delta t_1}$$
Expanding $e^{L\Delta t} = I + L\Delta t + \frac{(L\Delta t)^2}{2} + O(\Delta t^3)$:
$$\tilde{A} - A_{\Delta t_2} = O\left((L\Delta t_1)^2 \cdot \frac{\Delta t_2}{\Delta t_1}\right) - O\left((L\Delta t_2)^2\right) = O(\Delta t^2) \quad \blacksquare$$

### Corollary 5.8
*The continuous formulation provides exact temporal scale-free control; the discrete formulation requires relearning or higher-order corrections when changing control frequency.*

---

## 6. Why Discrete + DARE Works Despite Model Errors

### Theorem 6.1 (Certainty Equivalence)
*For linear-Gaussian systems with quadratic costs, the optimal policy separates estimation and control: $u^* = -K\hat{z}$ where $K$ is from Riccati and $\hat{z}$ is the state estimate.*

**Reference:** Anderson & Moore (1990), Chapter 8.

### Proposition 6.2 (Robustness to Eigenvalue Error)
*Let $L$ be the true generator and $\hat{L}$ the estimate with $\|L - \hat{L}\| < \epsilon$. The closed-loop system under LQR based on $\hat{L}$ remains stable if:*
$$\epsilon < \frac{\sigma_{min}(B)}{\|P\|}$$
*where $P$ is the Riccati solution.*

**Proof Sketch.** Small-gain theorem + perturbation analysis of Riccati equation. $\blacksquare$

### Remark 6.3 (Why "Wrong" Eigenvalues Work)
Control succeeds despite eigenvalue mismatch because:
1. LQR optimizes cost, not eigenvalue matching
2. Feedback continuously corrects deviations
3. High $Q$ (aggressive weighting) compensates for model errors
4. The qualitative structure (stable vs unstable directions) is usually preserved

---

## 7. Recommendations

### 7.1 When to Use Discrete + DARE
- Data sampled discretely, no model knowledge
- Fast controller execution relative to dynamics
- Robustness prioritized over interpretability

### 7.2 When to Use Continuous + CARE  
- Physics model available with analytical derivatives
- Multi-rate systems (different learning/actuation frequencies)
- Need spectral analysis for control design
- Coarse-graining robustness required

### 7.3 Hybrid Approach
1. **Learn** in discrete time (robust to noise)
2. **Convert** to continuous via matrix logarithm: $L = \log(A_d)/\Delta t$
3. **Analyze** spectrum for control frequency requirements
4. **Discretize** to actuation frequency via matrix exponential

---

## 8. Connection to Meyn's Framework

Sean Meyn's work on Markov chains, stochastic control, and reinforcement learning provides foundational theoretical grounding for data-driven control:

### 8.1 Key References

- **Meyn (2022)**: *Control Systems and Reinforcement Learning* — The definitive modern treatment unifying classical control with RL through the lens of Markov chain theory
- **Meyn & Tweedie (2009)**: *Markov Chains and Stochastic Stability* — Ergodic theory foundations

### Proposition 8.2 (Ergodic Justification for Learning)
*Under conditions ensuring ergodicity of the Markov chain induced by the controlled system, time averages converge to expectations:*
$$\frac{1}{N}\sum_{k=1}^N g(z_k) \xrightarrow{a.s.} \mathbb{E}_\pi[g(z)]$$
*This justifies learning from a single long trajectory.*

### Remark 8.3 (LQR as Special Case)
For linear systems with Gaussian noise, the ergodic measure $\pi$ is Gaussian with covariance satisfying a Lyapunov equation. Meyn's framework generalizes this to nonlinear systems.

### Remark 8.4 (Connection to EDMD)
Meyn's treatment of **temporal difference learning** (Chapter 7 of CSRL 2022) provides the probabilistic foundation for why EDMD's least-squares objective is statistically principled:
- The regression target $z_{k+1}$ is the one-step Bellman backup
- Regularization corresponds to prior beliefs (Bayesian interpretation)
- Convergence guarantees come from ergodicity of the exploration policy

### Definition 8.5 (Q-function and Koopman)
The **Q-function** $Q(x, u) = E[\sum_{k=0}^\infty \gamma^k c(x_k, u_k) | x_0 = x, u_0 = u]$ is an eigenfunction of the controlled Koopman operator (Meyn 2022, §3.5). This connects:
- Koopman eigenvalues ↔ discount rate / time scales
- Koopman eigenfunctions ↔ value functions
- EDMD ↔ Fitted Q-Iteration in function space

---

## 9. Conclusion

**Should we abandon kGEDMD + CARE?**

**No.** Both approaches are justified within their domains:

| Criterion | Discrete (DARE) | Continuous (CARE) |
|-----------|-----------------|-------------------|
| **Theoretical basis** | MLE/MAP | Galerkin projection |
| **Data requirements** | Transitions only | Derivatives or small $\Delta t$ |
| **Coarse-graining** | Requires rescaling | Scale-free |
| **Spectral insight** | Indirect | Direct |
| **Numerical robustness** | High | Moderate |

The continuous formulation provides crucial spectral information for control frequency selection that the discrete formulation obscures.

---

## References

1. Klus et al. (2020). "Data-driven approximation of the Koopman generator"
2. Williams et al. (2015). "A data-driven approximation of the Koopman operator"  
3. Kaiser et al. (2020). "Data-driven discovery of Koopman eigenfunctions for control"
4. Anderson & Moore (1990). *Optimal Control: Linear Quadratic Methods*
5. **Meyn (2022). *Control Systems and Reinforcement Learning*. Cambridge University Press.** — The modern unified treatment
6. Meyn & Tweedie (2009). *Markov Chains and Stochastic Stability*
7. Meyn (2007). *Control Techniques for Complex Networks*
8. Dellnitz & Junge (1999). "On the Approximation of Complicated Dynamical Behavior" — Set-oriented methods for coarse-graining
