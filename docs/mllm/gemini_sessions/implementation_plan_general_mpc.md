# Implementation Plan: Generalizable Koopman MPC

## Goal

Develop a **unified, problem-agnostic** control framework using Koopman operators and Model Predictive Control (MPC). This removes heuristic "swing-up" logic and provides a general solution for nonlinear systems.

## Core Philosophy

Instead of hardcoding control laws (like "pump energy"), we shift the burden to **optimization**.
1. **Model**: $z_{k+1} = A z_k + B u_k + N(z_k \otimes u_k)$ learned from data.
2. **Control**: MPC solves for the optimal $u_{k:k+H}$ that minimizes cost $J$.
3. **Generalization**: The *same code* works for any system where the Koopman model is accurate.

## Proposed Changes

### [NEW] `general_koopman_mpc.py`

#### 1. Generic Koopman Learner
- **Features**: Standard polynomial/RBF features (no manual selection).
- **Model**: Discrete Bilinear Koopman (proven robust in our tests).
- **Unsupervised Learning**: Learn from random exploration data.

#### 2. Linear/Bilinear MPC Controller
- **Objective**: Minimize error in lifted space $\sum \|z_k - z_{target}\|^2_Q + \|u_k\|^2_R$.
- **Constraints**: 
  - Dynamics: $z_{k+1} = Az_k + B_{eff}(z_k)u_k$
  - Actuation limits: $u_{min} \leq u \leq u_{max}$
- **Solver**: 
  - For linear features: Convex QP (fast).
  - For bilinear: Sequential QP (SQP) or dense shooting (if horizon is short).
  - *Recommendation*: Use **Convex MPC** with time-varying $B_k = B + N z_k$ (Linear Time-Varying LTV-MPC).

#### 3. Unified Validation
We will apply this **exact same class** to:
1. **Double Well**: Stabilize at origin (crossing saddle).
2. **CartPole**: Swing-up and stabilize (pumping energy).

## Verification Plan

### Experiment 1: Double Well
- Run `GeneralKoopmanMPC` on Double Well.
- Verify it stabilizes at the saddle point without DARE/CARE tuning.

### Experiment 2: CartPole Swing-Up
- Run `GeneralKoopmanMPC` on CartPole.
- Verify it **automatically discovers** the pump-and-stabilize behavior.
- *Success Criteria*: Reaches upright state without any "energy" code.

## Why This Is Better
1. **No Physics Knowledge**: Doesn't know what "energy" or "gravity" is.
2. **No Mode Switching**: Single controller for swing-up and stabilization.
3. **Scalable**: Works for higher-dim systems by just changing `env`.

---

## Technical Details

**LTV-MPC Formulation**:
At each step $t$:
1. Predict trajectory using constant input guess.
2. Linearize bilinear term around prediction: $z_{k+1} \approx A z_k + (B + N \bar{z}_k) u_k$.
3. Solve convex QP for $u$.
4. Apply $u_0$.

**Horizon**: $H=20$ to 50 steps (lookahead needed to see "benefits" of pumping).
