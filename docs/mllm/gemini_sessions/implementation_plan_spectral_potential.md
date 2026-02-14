# Implementation Plan: RKHS-KRONIC Spectral Potential Control

## Goal

Achieve CartPole swing-up and stabilization using a **single, unified control law** derived purely from data-driven Koopman eigenfunctions. Eliminate ad-hoc physics-based switching logic.

## Theoretical Basis

1.  **Spectral Decomposition**: The system dynamics are decomposed into Koopman modes.
    -   **Continuous Spectrum** ($|\lambda| \approx 1$): Corresponds to energy/conserved quantities.
    -   **Discrete Spectrum** ($|\lambda| < 1$, e.g., $\lambda \approx 0.9$): Corresponds to the stable manifold of the attractive fixed point (upright).

2.  **Spectral Potential**: We define a "target potential" in the eigenfunction space:
    $$V(z) = \frac{1}{2} w_E (\phi_E(z) - \phi_E^{target})^2 + \frac{1}{2} w_S \phi_S(z)^2$$
    -   Term 1 drives system to the correct energy shell (Swing-Up).
    -   Term 2 drives system along the stable manifold to the fixed point (Stabilization).

3.  **Control Law**:
    $$u = -k \cdot \text{sign}\left( \frac{\partial V}{\partial u} \right)$$
    The gradient $\frac{\partial V}{\partial u}$ is computed via the Chain Rule through the kernel expansion.

## Proposed Changes

### [MODIFY] `poc_rkhs_energy_control.py`

#### 1. Discover Stability Mode
- Search for an eigenfunction $\phi_S$ with eigenvalue $|\lambda| \in [0.8, 0.95]$.
- Select the one that has high variance near the upright state but decays away from it (or vice-versa).

#### 2. Unified Controller
- Remove `LinearKoopmanLQR` class and `if/else` logic.
- Implement `evaluate_spectral_potential_grad(x)`:
  - Compute $\phi_E, \phi_S$.
  - Compute $\nabla \phi_E, \nabla \phi_S$.
  - Combine: $\nabla V = w_E (\phi_E - \phi_E^*) \nabla \phi_E + w_S \phi_S \nabla \phi_S$.

#### 3. Tuning
- Balance weights $w_E$ (Energy) vs $w_S$ (Stability). Likely $w_E \gg w_S$ initially, but near upright $\phi_E \approx \phi_E^*$ so $\phi_S$ dominates. This gives **endogenous switching**.

## Verification Plan

### Experiment
- Run `poc_rkhs_energy_control.py`.
- Verify the controller swings up (driven by $\phi_E$).
- Verify it stabilizes at top (driven by $\phi_S$ locking in phase).
- Plot contributions of $\phi_E$ and $\phi_S$ to the control signal to visualize the "soft switch".
