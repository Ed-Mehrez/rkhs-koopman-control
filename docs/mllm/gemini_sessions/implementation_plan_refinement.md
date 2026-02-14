# Principled Refinement of RKHS CartPole Control

The objective is to eliminate the "Propeller Effect" (unwanted spinning of the pole during drift) and ensure a stable, principled swing-up and stabilization of the CartPole system.

## User Review Required

> [!IMPORTANT]
> **State Augmentation Change**: We will formally replace the `Raw Theta` state augmentation with a `Bounded Manifold Augmentation` using $[\cos\theta, \sin\theta]$. This is a fundamental change to the linear state feedback structure used by LQR.

> [!NOTE]
> **Drift Acceptance**: The plan accepts that without a finite rail, the cart will drift. The goal is to make this drift "Gentle" (constant velocity, free spin) rather than "Violent" (reaction mass propulsion).

## Proposed Changes

### 1. Bounded State Augmentation
Modified `poc_rkhs_cartpole.py`:
-   **Current**: `get_features` augments Kernel features `$z_k$` with `Raw State` $[x, \dot{x}, \theta, \dot{\theta}]$.
-   **New**: `get_features` will augment with $[x, \dot{x}, \cos\theta, \sin\theta, \dot{\theta}]$.
-   **Rationale**: This removes the unbounded $\theta$ variable from the linear state seen by LQR, preventing large linear errors from driving the system when the angle wraps. It embeds the periodic state into a bounded manifold where Euclidean distance corresponds to the true chordal distance on the circle.

### 2. Precise Target State
Modified `poc_rkhs_cartpole.py`:
-   Ensure `z_target` corresponds exactly to the Augmented State of the "Upright" position:
    -   $x=0, \dot{x}=0$
    -   $\cos\theta = 1, \sin\theta = 0$
    -   $\dot{\theta} = 0$
-   Verify `Fixed Point Error` of the learned model at this target. If significant ($>0.1$), consider solving for the nearest dynamical equilibrium.

### 3. Aggressive Tuning (Stabilization Mode)
Adjust LQR weights (`Q` and `R`) to enforce the "Locking" behavior:
-   **$Q_{angle}$ (Cos/Sin)**: Set to `100.0`. High penalty for deviation from upright.
-   **$Q_{rate}$ (Theta_dot)**: Set to `20.0`. Moderate damping to prevent oscillations.
-   **$Q_{position}$ (x)**: Set to `0.01` (or similarly low value).
    -   **Explanation**: Decouples position error from stabilization. Tells the controller "Do not sacrifice angle stability to fix $x$".
-   **$R$**: Set to `2.0`. Balanced effort.

## Verification Plan

### Automated Simulation
-   Run `poc_rkhs_cartpole.py` with the new configuration.
-   Capture simulation logs.

### Analysis Criteria
-   **Propeller Effect**: Must be eliminated. Defined as "Continuous rapid rotation of the pole".
-   **Stability**: Angle $\theta$ must remain near $2k\pi$ (Upright) once swung up.
-   **Control Effort**: $U$ should settle to near zero (or small equilibrium value) once upright, not oscillate wildly.
-   **Drift**: Accepted (Linear drift is fine, exponential acceleration is bad).

### Manual Verification
-   **Artifact**: `rkhs_cartpole_swingup_sdre.gif`
-   **Check**: Visually confirm the pole swings up and then *holds* (or wobbles gently) while the cart travels. NO violent spinning.
