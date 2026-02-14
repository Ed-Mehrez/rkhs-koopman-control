# Implementation Plan: RKHS-KRONIC Energy Control

## Goal

Implement a **pure Kernel-based controller** for CartPole swing-up and stabilization.
- **No MPC**: No runtime optimization.
- **No Physics**: No $E = mgh$ formulas.
- **Infinite Dimensions**: Use RBF kernels to handle the continuous spectrum.

## Theoretical Basis

1. **Kernel Trick**: We work with the Gram matrix $G_{ij} = k(x_i, x_j)$ to avoid finite basis truncation issues.
2. **Continuous Spectrum**: The accumulation of eigenvalues $\lambda_i \to 1$ on the unit circle represents the continuous spectrum of the Hamiltonian dynamics.
3. **Invariance Principle**: The eigenvector $v$ associated with $\lambda \approx 1$ defines an invariant level set function $\phi(x) = \sum \alpha_i k(x_i, x)$.
4. **Lie Derivative Control**: We want to change the value of $\phi(x)$ (energy) to reach the target level.
   $$\dot{\phi} = \nabla \phi \cdot (f + g u) = L_f \phi + u L_g \phi$$
   Control law: $u = -k \cdot \text{sign}(L_g \phi) \cdot (\phi - \phi_{target})$

## Proposed Changes

### [NEW] `poc_rkhs_energy_control.py`

#### 1. Kernel Koopman Solver
- Compute Gram matrices $G = K(X, X)$ and $A = K(X, Y)$.
- Solve generalized eigenproblem $A V = G V \Lambda$ (Kernel EDMD).
- **Result**: Eigenfunctions expansion coefficients $\alpha$.

#### 2. Energy Eigenfunction Discovery
- Automatically select the eigenfunction $\phi$ that:
  - Has eigenvalue $|\lambda| \approx 1$.
  - Has non-trivial variance (not the constant mode).
  - Correlates with state magnitude (optional check).

#### 3. RKHS Controller
- Implement the function $\phi(x) = \sum_{i=1}^N \alpha_i k(x_i, x)$.
- Implement the gradient $\nabla \phi(x) = \sum \alpha_i \nabla_x k(x_i, x)$.
- **Control Law**:
  $$u(x) = -K_{gain} \cdot (\phi(x) - \phi_{upright}) \cdot \text{sign}(\nabla \phi \cdot \hat{g})$$
  where $\hat{g} = [0, 1, 0, 0]^T$ (approximate actuation direction).

## Verification Plan

### Experiment: CartPole Swing-Up
1. **Unsupervised Learning**: Collect random trajectories.
2. **Eigenfunction Plot**: Visualize $\phi(x)$ vs True Energy to confirm discovery.
3. **Closed-Loop Sim**: Run the RKHS feedback controller. 

## Computational Considerations
- **Nystrom Approximation**: Kernel methods scale as $O(N^3)$. We will use Nystrom sampling (M centers) to keep it fast.
- **Centers**: Choose $M=500$ centers using k-means clustering on the data.

## Why This Fits "KRONIC Style"
- Uses RKHS norm and inner products.
- Handles the "infinite dimensional representation" requirement.
- Control is a functional in the RKHS.
