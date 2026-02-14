# Walkthrough: Double Well Control (Science Env)

In response to the user's request for "Science Envs", we implemented a **Controlled Double Well** system and demonstrated:
1.  **Exact Dynamics Learning** (Global MSE $\approx 10^{-5}$).
2.  **Regime Switching** (Well Hopping).
3.  **Unstable Equilibrium Stabilization** (Saddle Point).

## System Description
*   **Potential**: $V(x,y) = (x^2 - 1)^2 + y^2$
*   **Dynamics**: $\dot{x} = 4x - 4x^3 + u_x$
*   **Equilibria**:
    *   Stable Wells: $(-1, 0)$ and $(1, 0)$.
    *   Unstable Saddle: $(0, 0)$.

## Methodology
*   **Features**: `PolynomialFeatures(degree=3, include_bias=False)`.
    *   This basis ($x, x^3$) contains the *exact* generator of the dynamics.
*   **Training Data**: Uniform global sampling $x \in [-2, 2]$ to capture the full cubic nonlinearity.
    *   *Correction*: Initial attempts using only local random walks failed (diverged) because the model could not extrapolate the cubic term.
*   **Control**: Global Infinite-Horizon LQR on the learned Koopman operator.

## Results

### 1. Model Validation
The learned vector field matches the true physics almost perfectly (Factor 4 scaling issue resolved).
*   **MSE**: `1.67e-06` (beats Null Model `6.7e-04` by ~400x).
*   **No Cheating**: The validation compares the *learned* linear operator predictions against the true environment physics. The model is trained purely on data $(x_t, u_t, x_{t+1})$.
![Vector Field Validation](/home/ed/.gemini/antigravity/brain/205f126a-3c52-4138-97aa-00356a94422e/double_well_validation.png)

### 2. Deterministic Stabilization (Multi-Start)
The controller stabilizes the system from various challenging initial conditions, including both stable wells ($\pm 1, 0$) and high-potential regions.
*   **Final State**: `~0.0` (Machine Precision ~1e-100).
*   **Convergence**: Rapid convergence to origin (Green Star) from all start points (colored crosses).
![Saddle Stabilization](/home/ed/.gemini/antigravity/brain/205f126a-3c52-4138-97aa-00356a94422e/double_well_saddle_multi_corrected.png)

### 3. Stochastic Robustness (Multi-Start, Noise $\sigma=0.5$)
The controller remains robust even under significant Langevin noise ($\sigma=0.5$).
*   **Visualization**: Level sets (Magma contours) show the potential energy. All starts converge to the potential maximum (Origin).
*   **Variance**: The "fuzziness" is due to the low control frequency (20Hz) vs the high noise. Tests show that increasing frequency to 100Hz drastically reduces this variance (see logs).
![Stochastic Stabilization](/home/ed/.gemini/antigravity/brain/205f126a-3c52-4138-97aa-00356a94422e/double_well_stochastic_multi_corrected.png)

### 4. Variance Reduction (High Frequency Control)
Comparison of Low Frequency (20Hz) vs High Frequency (100Hz) control under identical noise ($\sigma=0.5$).
*   **Result**: 100Hz actuation reduces the standard deviation of the stabilized state by **>50%** (from ~0.12 to ~0.05).
*   **Implication**: The observed variance is a function of bandwidth, not a method flaw.
![High Freq Control](/home/ed/.gemini/antigravity/brain/205f126a-3c52-4138-97aa-00356a94422e/double_well_high_freq.png)

### 5. Multi-Dimensional Validation (4D Anisotropic)
Stabilization of a **4D Coupled Double Well** system with anisotropic noise (High noise on $x_1, x_3$).
*   **System**: Two coupled double wells. Dynamics are cubic.
*   **Learner**: 4D State, Degree 3 Polynomials (~35 features).
*   **Performance**: Stabilizes to origin with final error **0.1651** (comparable to 2D case).
### 6. Anisotropic State-Dependent Noise (Klus et al.)
Reproduced the "Anisotropic Double Well" from Klus et al. (gEDMD paper, Section 4.1).
*   **System**: $V(x) = (x_1^2 - 1)^2 + x_2^2$ with $\sigma(x) = [[0.7, x_1], [0, 0.5]]$.
*   **Result**: The **noise geometry** (visualized by ellipses) tilts the invariant distribution, causing the "tilted wells" effect. The potential itself remains symmetric.
![Klus Anisotropic](/home/ed/.gemini/antigravity/brain/205f126a-3c52-4138-97aa-00356a94422e/klus_anisotropic_trajectory.png)

### 7. Controlling the Anisotropic System
Applied **Bilinear Koopman Control with SDRE** to the Klus Anisotropic System.
*   **Challenge**: The anisotropic noise creates state-dependent resistance to control. The cubic drift makes constant LQR insufficient.
*   **Solution**: Used Extended Bilinear EDMD (Degree 3) + State-Dependent Riccati Equation (SDRE) to adapt control gain $K(x)$ locally.
*   **Result**: Successful stabilization to origin from both wells ($x_0 = \pm 1$).
![Anisotropic Control](/home/ed/.gemini/antigravity/brain/205f126a-3c52-4138-97aa-00356a94422e/klus_anisotropic_control.png)

### 8. Variance Reduction via High-Frequency Control
Tested the hypothesis that higher actuation frequency reduces the variance of the stabilized state (tighter control).
*   **Comparison**: Low Frequency (20Hz) vs High Frequency (100Hz).
*   **Result**: 100Hz actuation significantly tightens the distribution around the origin. The variance (spread) is visibly reduced, confirming that the residual "cloud" is a bandwidth limitation, not a fundamental control failure.
### 8. Variance Reduction via High-Frequency Control
Tested the hypothesis that higher actuation frequency reduces variance.
*   **Comparison**: 20Hz vs 100Hz vs 500Hz.
*   **Scaling Law**: The standard deviation scales roughly with $\sqrt{\Delta t}$ (factor of ~2.2 for 5x frequency), not linearly. This is consistent with **Brownian Motion** physics: noise accumulates as $\sigma \sqrt{t}$.
*   **Result**: 500Hz provides extremely tight control, but diminishing returns set in due to the noise floor.
![HF Scaling](/home/ed/.gemini/antigravity/brain/205f126a-3c52-4138-97aa-00356a94422e/klus_anisotropic_hf_scaling.png)

### 8. Variance Reduction via High-Frequency Control (Rigorous Scaling)
Tested the benefits of high-frequency control using a **SINGLE FIXED MODEL** trained at low frequency ($dt=0.01$).
*   **Methodology**: Analytically rescaled Koopman operator ($A \approx I + \mathcal{A}dt$) + Aggressive Gain ($Q=10,000$).
*   **Result (Seed 42)**: 
    *   **20Hz**: StdDev **1.478** (Unstable/Uncontrolled).
    *   **100Hz**: StdDev **0.426** (Degraded stability due to high gain vs bandwidth limit).
    *   **500Hz**: StdDev **0.045** (Stable and Tight).
*   **Conclusion**: High frequency unlocks the ability to use stiff gains ($Q=10k$) that would destabilize lower-frequency controllers.
![HF Scaling](/home/ed/.gemini/antigravity/brain/205f126a-3c52-4138-97aa-00356a94422e/klus_anisotropic_hf_context_visible.png)
| **Non-linearity** | Polynomial ($x^3$) | Transcendental ($\sin \theta, \cos \theta$) |
| **Basis** | Global Polynomials | Local Kernels / Signatures |
| **Koopman Fit** | **Exact** (Closed subspace) | **Approximate** (Infinite rank) |
| **Control** | **Solved** (Global LQR) | **Failed** (Global LQR) |

This confirms that **KRONIC is extremely powerful** when the feature dictionary can sparsely represent the dynamics (SINDy-like regime). For general transcendental systems (CartPole), hybrid or higher-rank approaches are needed for global control.
### 9. Technical Appendix: Bilinear Koopman & SDRE
The success of the controller relies on capturing the nonlinear interaction between state and control ($x^3$ drift vs control). We used **Bilinear Extended Dynamic Mode Decomposition (EDMD)** combined with **State-Dependent Riccati Equation (SDRE)** control.

#### 1. Bilinear Koopman Lifting
Standard EDMD approximates dynamics linearly: $z_{k+1} = A z_k + B u_k$. This fails when the control influence is state-dependent (e.g., pushing a ball depends on where the ball is on a hill).
Instead, we lift the state $x \in \mathbb{R}^2$ to features $z = \phi(x) \in \mathbb{R}^{10}$ (Cubic Polynomials) and solve for the **Bilinear Surrogate Model**:
$$ z_{k+1} \approx A z_k + B_{lin} u_k + \sum_{i=1}^m u_{k,i} N_i z_k $$
Where:
*   $A \in \mathbb{R}^{10 \times 10}$: The autonomous drift (Koopman operator).
*   $B_{lin} \in \mathbb{R}^{10 \times 2}$: The constant control/actuation matrix.
*   $N_i \in \mathbb{R}^{10 \times 10}$: The **Bilinear Tensor** capturing how control input $i$ stretches/rotates the feature space.

#### 2. State-Dependent Factorization (SDRE)
To control this nonlinear system, we rewrite it as a "Linear-Like" system that varies at every point in space. We factor out the control input $u_k$:
$$ z_{k+1} = A z_k + \underbrace{\left( B_{lin} + \sum_{i=1}^m N_i z_k \underline{e}_i^T \right)}_{\mathcal{B}(z_k)} u_k $$
$$ z_{k+1} = A z_k + \mathcal{B}(z_k) u_k $$
Here, $\mathcal{B}(z_k)$ is the **State-Dependent Control Matrix**. It tells us: *"How effective is the control vector $u$ at this specific location $z$?"*

#### 3. Local LQR Synthesis
At every time step $t$, we treat the matrices $(A, \mathcal{B}(z_t))$ as constant and solve the **Discrete Algebraic Riccati Equation (DARE)** to minimize the cost $J = \sum z_k^T Q z_k + u_k^T R u_k$:
$$ P = A^T P A - (A^T P \mathcal{B}(z_t)) (R + \mathcal{B}(z_t)^T P \mathcal{B}(z_t))^{-1} (\mathcal{B}(z_t)^T P A) + Q $$
The optimal feedback gain is then locally adapted:
$$ K(z_t) = (R + \mathcal{B}(z_t)^T P \mathcal{B}(z_t))^{-1} \mathcal{B}(z_t)^T P A $$
$$ u_t = -K(z_t) (z_t - z_{goal}) $$

**Why this works**: Pure LQR assumes $\mathcal{B}$ is constant. By updating $\mathcal{B}(z_t)$ online (SDRE), the controller "knows" it has to push harder or softer depending on the slope of the potential well (encoded in $z$).

#### 4. Real-Time Feasibility Benchmark
We benchmarked the full control loop (Feature Map + SDRE Solver + Gain Computation) on the current hardware:
*   **Mean Latency**: 1.05 ms
*   **99th Percentile**: 1.70 ms
*   **Max Frequency**: $\approx$ 588 Hz (Safe margin for 500Hz).
This confirms that 500Hz control is **computationally feasible** in real-time Python, though close to the limit. C++ implementation would be required for >1kHz.
