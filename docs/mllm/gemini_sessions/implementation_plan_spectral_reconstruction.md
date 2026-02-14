# Implementation Plan: General Spectral Reconstruction Control

## Goal

Create a **general-purpose** controller that solves nonlinear tasks by minimizing a quadratic cost on the **reconstructed state** from Koopman eigenfunctions. This removes the need to manually identify "energy" or "stability" modes or assign ad-hoc weights.

## Core Logic: `SpectralReconstructionController`

1.  **Learn**: Kernel EDMD on unforced/random data.
2.  **Reconstruct**: Learn the projection vectors (Koopman Modes) $v_k$ by solving the least-squares problem:
    $$ X \approx \Phi(X) V $$
    where $\Phi(X)$ is the matrix of eigenfunctions.
3.  **Cost Construction**:
    Lift the physical cost $J = x^T Q x$ to the spectral space:
    $$ J(z) = (\Phi(z) V)^T Q (\Phi(z) V) $$
    This automates the weighting: modes that contribute significantly to the state error (large trajectory excursions) get weighted heavily during swing-up, while local modes dominate near the origin.
4.  **Control**:
    $$ u = -k \cdot \text{sign}(\nabla_u J) $$
    $$ \nabla_u J = 2 (\Phi(z) V)^T Q (V^T \nabla_u \Phi(z)) $$

## Validation Experiments

### 1. Unified Script `general_spectral_control.py`
Run the **same controller class** on:
- **Double Well**: Target $x^*=[0,0]$. Q=Identity.
- **CartPole**: Target $x^*=[0,0,\text{upright},0]$. Q=Diag([0,1,10,1]).

## Why This is General
- It recovers LQR locally.
- It recovers "Energy Pumping" globally (since energy modes contribute to large state excursions).
- It requires no "mode selection" logic, just the standard $Q$ matrix description of the task.
