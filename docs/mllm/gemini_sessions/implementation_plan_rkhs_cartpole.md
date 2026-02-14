# Implementation Plan - RKHS-KRONIC CartPole Swing-Up

## Goal
Demonstrate that the **Pure Kernel** approach (Kernel EDMD with Gaussian RBF) can solve the **CartPole Swing-Up** problem.
This is a benchmark that **Polynomial** features failed at (due to periodicity/global nonlinearity).
We expect RBF kernels (universal approximators) to succeed where polynomials failed.

## User Review Required
> [!IMPORTANT]
> **Periodicity**: CartPole state is $[x, \dot{x}, \theta, \dot{\theta}]$. $\theta$ is periodic. The Gaussian Kernel $k(x, y) = \exp(-\|x-y\|^2)$ is *not* intrinsically periodic, but with enough centers (data), it can approximate the periodic topology.
> **Alternative**: We could use a **Periodic Kernel** $k(\theta, \theta') = \exp(-\sin^2(\theta-\theta'))$, but for now we test the "Black Box" power of the standard RBF.

## Proposed Changes

### 1. New Example Script
#### [NEW] [poc_rkhs_cartpole.py](file:///home/ed/SynologyDrive/Documents/Research/P&E_Research/rkhs_kronic/examples/proof_of_concept/poc_rkhs_cartpole.py)
*   Integrates `src.kedmd_core.KernelEDMD` with the CartPole environment.
*   **Data Collection**: Use **Uniform Random Sampling** (as verified in PoC 10) over the full phase space:
    *   $x \in [-2, 2]$
    *   $\theta \in [-\pi, \pi]$ (Critical for coverage)
    *   $\dot{x}, \dot{\theta}$ (velocities)
*   **Kernel**: Gaussian RBF with Median Heuristic + `StandardScaler`.
*   **Control**: SDRE (State-Dependent Riccati Equation) using Analytic Gradients.

### 2. Visualization
*   Compare Swing-Up performance against:
    *   Linear LQR (fails)
    *   Previous Polynomial Methods (failed swing-up)
*   Plot Energy shaping ($E \to E_{target}$).

## Verification Plan
### Automated Tests
*   Run the script `python examples/proof_of_concept/poc_rkhs_cartpole.py`.
*   Success Criteria:
    *   CartPole swings up from $\theta = \pi$ (down) to $\theta = 0$ (up).
    *   Stabilizes at top.
    *   Reward > 300 (standard Gym metric).

### Manual Verification
*   Inspect the generated plot `rkhs_cartpole_swingup.png`.
*   Verify that the "Gap" in the trajectory (due to $\theta$ wrap-around) is handled smoothly.
