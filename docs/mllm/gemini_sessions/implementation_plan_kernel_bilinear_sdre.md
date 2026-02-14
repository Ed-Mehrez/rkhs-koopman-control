# Consolidated Implementation Plan: Kernel Bilinear SDRE Control

## Goal

Create a **unified, principled, general-purpose** Koopman control framework that:
1. Uses the **bilinear Koopman** formulation (proven to work in `poc_klus_2d_control.py`)
2. Operates in **kernel space** using Gram matrices (infinite dimensions, no truncation)
3. Uses **SDRE** with the learned state-dependent control matrix
4. Avoids problem-specific heuristics

---

## Background: What Worked Before

### `poc_klus_2d_control.py` (Double Well - SUCCESS)
- **Formulation**: $z_{k+1} = A z_k + B_{lin} u_k + \sum_i u_i B_i z_k$
- **Features**: Polynomial degree 3 (finite, explicit)
- **Control**: Discrete ARE (DARE) with $B_{eff}(z) = B_{lin} + \sum_i B_i z$
- **Result**: Successfully stabilizes at saddle point

### What Failed Recently
- **Spectral Potential Control**: Sign ambiguities in gradients, wrong direction
- **State Reconstruction**: Local minima, vanishing gradients
- **MPC**: User rejected as "cheating" (not KRONIC-style)

---

## Proposed Architecture: Kernel Bilinear SDRE

### Core Idea
Instead of explicit polynomial features $\phi(x)$, use the **kernel trick**:
$$\phi(x)^T \phi(y) = k(x, y)$$

This allows infinite-dimensional features while only computing Gram matrices.

### Bilinear Learning in Kernel Space

**Goal**: Learn the bilinear dynamics:
$$\Psi(x_{k+1}) = \mathcal{A} \Psi(x_k) + \mathcal{B}_{lin} u_k + \sum_j u_j \mathcal{B}_j \Psi(x_k)$$

**Method**: Extended Kernel EDMD
1. Compute Gram matrices $G = K(X, X)$ and $G' = K(X', X)$
2. Regressor: $\mathcal{R} = [G, U, G \odot u_1, G \odot u_2, ...]$  (Hadamard products for bilinear terms)
3. Solve: $\mathcal{R}^T \mathcal{R} \theta = \mathcal{R}^T G'$
4. Extract $\mathcal{A}, \mathcal{B}_{lin}, \mathcal{B}_j$ from $\theta$

### Control via DARE in Kernel Space

At each state $x$:
1. **Compute kernel vector**: $\psi(x) = [k(x, x_1), ..., k(x, x_N)]^T$
2. **Compute effective B**:
   $$\mathcal{B}_{eff}(\psi) = \mathcal{B}_{lin} + \sum_j u_j^{prev} \mathcal{B}_j$$
3. **Solve Discrete ARE**: For $(\mathcal{A}, \mathcal{B}_{eff}, Q, R)$
4. **Compute gain**: $K = (R + \mathcal{B}_{eff}^T P \mathcal{B}_{eff})^{-1} \mathcal{B}_{eff}^T P \mathcal{A}$
5. **Apply**: $u = -K \psi(x)$

### Analytic Kernel Gradients (KRONIC-style)

For Lie derivative control (alternative to SDRE):
$$L_g \phi = \nabla \phi \cdot g = \sum_i \alpha_i \nabla k(x_i, x) \cdot g$$

Using `RBFKernel.diff()` from `kgedmd_core.py`:
$$\nabla k(x_i, x) = -\frac{x - x_i}{\sigma^2} k(x_i, x)$$

---

## Proposed Changes

### [NEW] `examples/proof_of_concept/kernel_bilinear_sdre.py`

```python
class KernelBilinearKoopman:
    def __init__(self, kernel, reg=1e-5):
        self.kernel = kernel
        self.reg = reg
        
    def fit(self, X, U, X_next):
        # 1. Compute Gram matrices
        G = self.kernel(X, X)      # (N, N)
        G_prime = self.kernel(X_next, X)  # (N, N)
        
        # 2. Build regressor [G, U, G*u1, G*u2, ...]
        n_samples, action_dim = U.shape
        regressors = [G, U]
        for i in range(action_dim):
            regressors.append(G * U[:, i:i+1])  # Hadamard
        R = np.hstack(regressors)
        
        # 3. Ridge regression
        theta = solve(R.T @ R + self.reg * I, R.T @ G_prime)
        
        # 4. Parse theta into A, B_lin, B_bilin
        self.A = theta[:N, :].T
        self.B_lin = theta[N:N+action_dim, :].T
        self.B_tensor = [theta[idx:idx+N, :].T for ...]
        
    def get_B_eff(self, psi):
        B = self.B_lin.copy()
        for i, Bi in enumerate(self.B_tensor):
            B += Bi @ psi.reshape(-1, 1) * scale
        return B

class KSDRE:
    def solve(self, psi, A, B_eff, Q, R):
        P = solve_discrete_are(A, B_eff, Q, R)
        K = solve(R + B_eff.T @ P @ B_eff, B_eff.T @ P @ A)
        return -K @ psi
```

---

## Verification Plan

### Test 1: Double Well Saddle Stabilization
- **Command**: `python examples/proof_of_concept/kernel_bilinear_sdre.py --env double_well`
- **Expected**: System moves from $x=[-1.5, 0]$ to $x=[0, 0]$ (saddle)
- **Baseline**: Compare with `poc_klus_2d_control.py` (polynomial features)

### Test 2: CartPole Swing-Up
- **Command**: `python examples/proof_of_concept/kernel_bilinear_sdre.py --env cartpole`
- **Expected**: Pendulum swings up from $\theta=\pi$ to $\theta=0$ and stabilizes
- **Metric**: Settling time, control effort

### Manual Verification
1. Run the script and observe printed trajectory
2. Verify error decreases monotonically (or at least converges)
3. Check that the same code works for both systems without modification

---

## Why This is General

1. **No explicit features**: Kernel trick handles infinite dimensions
2. **No hardcoded physics**: B matrix learned entirely from data
3. **Unified control**: SDRE works for both swing-up (global) and stabilization (local)
4. **Proven formulation**: Bilinear Koopman + DARE is mathematically rigorous

---

## Comparison with Previous Attempts

| Approach | Problem | Fix in This Plan |
|----------|---------|------------------|
| Spectral Potential | Sign ambiguity | Use SDRE (no sign issues) |
| State Reconstruction | Local minima | Direct control via DARE |
| MPC | "Cheating" | SDRE is feedback policy, not optimization |
| Nystrom Approx | Sparse coverage | Full Gram matrix (N x N) |
