# Walkthrough: RKHS Kernel Stabilization for CartPole

## Goal
Achieve stable control of the CartPole system using an Infinite-Dimensional Kernel (RBF/Periodic) without succumbing to phantom drift or spurious instabilities.

## Challenges Identified
1.  **Phantom Model Drift**: The learned model predicted a non-zero velocity ($A z_{target} \neq 0$) at the equilibrium point, causing the controller to fight a "phantom model force".
2.  **Spurious Instabilities**: The RBF Kernel is highly flexible and fits noise as high-frequency unstable modes. We observed 76 unstable eigenvalues, whereas the physics has only 1 (Gravity).
3.  **Regularization Sensitivity**: Tuning the regularization parameter ($\alpha$) proved insufficient to separate noise from physics.

## Solution Implemented

### 1. Strict Feature Centering (Drift Elimination)
We shifted the feature map origin to the equilibrium point:
$$\phi'(x) = \phi(x) - \phi(x_{eq})$$
This guarantees that $\phi'(x_{eq}) = \vec{0}$, ensuring $A \phi'(x_{eq}) = \vec{0}$ by construction.
**Result:** Drift metric reduced from `43.5` to `0.000000`.

### 2. Physics-Informed Spectral Stabilization (vs Regularization)
We conducted a parameter sweep of Ridge Regularization ($\alpha$) vs Spectral Clamping:

| Approach | Alpha | Top Unstable $\lambda$ (Real) | Count (Unstable) | Stability | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | $10^{-6}$ | ~20.0 | 76 | Unstable (0.1s) | Explodes (Overfit) |
| **High Reg** | $10^{-2}$ | ~0.34 | 6 | Unstable (Falls) | Over-damped (Weak K) |
| **Med Reg** | $10^{-3}$ | ~0.68 | 9 | Unstable (Falls) | Over-damped |
| **Low Reg** | $10^{-4}$ | ~12.1 | 17 | Unstable (Exp) | Under-damped |
| **Pure Tuning**| $8 \times 10^{-4}$ | ~1.0 | 1 | Unstable (Falls) | No "Goldilocks" zone |
| **Clamping** | $10^{-6}$ | **4.0 (Clamped)** | **1** | **Stable (>5s)** | **Optimal** |

**Conclusion:** Regularization is a "blunt instrument" that blunts the physical gravity mode ($\lambda \approx 3.1$) before it fully suppresses high-frequency noise.
**Strategy:** We use minimal regularization ($\alpha=10^{-6}$) to preserve dynamics sharpness, and explicitly **Clamp** the dominant unstable eigenvalue to $\lambda=4.0$ while damping all others.

### 3. Length Scale Smoothing
We relaxed the kernel `length_scale` from `1.0` to `4.0` to promote smoother, polynomial-like generalization.

## Verification Results
- **Stability:** The system balances for >5 seconds (1000 steps). Instability only occurs when the cart travels outside the training support ($x > 10$).
- **Control Effort:** Control gains reduced from spurious $10^8$ to realistic $\approx 240$.

![Comparison of Control Frequencies (Stabilization)](/home/ed/SynologyDrive/Documents/Research/P&E_Research/rkhs_kronic/rkhs_cartpole_freq_compare.png)

## Key Code Changes

render_diffs(file:///home/ed/SynologyDrive/Documents/Research/P&E_Research/rkhs_kronic/examples/proof_of_concept/poc_rkhs_cartpole.py)
