# Implementation Plan: General Spectral Potential Control

## Goal

Create a **general-purpose** controller that solves nonlinear tasks (e.g., Double Well saddle crossing, CartPole swing-up) by minimizing a **Spectral Potential** constructed from automatically discovered Koopman eigenfunctions.

## Core Logic: `SpectralPotentialController`

1.  **Learn**: Kernel EDMD on unforced/random data.
2.  **Classify Modes**:
    -   **Energy Proxy ($\phi_E$)**: Eigenvalue closest to 1, high variance.
    -   **Stability Proxy ($\phi_S$)**: Eigenvalue in $[0.9, 0.99]$, maximum correlation with distance to target $x^*$.
3.  **Construct Potential**:
    $$V(x) = w_E (\phi_E(x) - \phi_E(x^*))^2 + w_S (\phi_S(x) - \phi_S(x^*))^2$$
4.  **Control**:
    $$u = -k \cdot \text{sign}(\nabla V \cdot g)$$
    where $g$ is the actuation direction (learned or assumed known).

## Validation Experiments

### 1. Double Well (Saddle Stabilization)
- **Task**: Move from one well to the unstable saddle point.
- **Expectation**: $\phi_E$ captures the "energy" required to climb the potential hill. $\phi_S$ captures the stable manifold of the saddle.

### 2. CartPole (Swing-Up)
- **Task**: Move from bottom to unstable upright.
- **Expectation**: $\phi_E$ captures mechanical energy. $\phi_S$ captures local attraction to upright.

## Artifacts
- **File**: `examples/proof_of_concept/general_spectral_potential.py`
- **Output**: Phase plots for both systems showing convergence.

## Why This is General
It relies ONLY on spectral properties of the dynamical operator, which exists for *any* system. It translates "control goals" into "spectral geometry".
