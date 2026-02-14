# Implementation Plan - Spectral Bandwidth Selection

We will implement **Spectral Analysis** to analytically determine the optimal window size for Sig-RLS, replacing heuristics/autotuning with a physics-based derivation.

## Concept
The observed process is $X_t = \text{Drift}_t + \text{Diffusion}_t$.
*   **Drift (Signal):** Low-frequency "Red Noise" ($1/f^2$ spectrum).
*   **Diffusion (Noise):** Wide-band "White Noise" (Flat spectrum).

The **Optimal Window** corresponds to the time-scale $T = 1/f_c$, where $f_c$ is the **Corner Frequency** where the Signal PSD drops below the Noise Floor.

## Proposed Changes

### [NEW] `examples/proof_of_concept/poc_spectral_bandwidth.py`

This script will:
1.  **Simulate:** The Regime Switching OU process.
2.  **Spectral Analysis:**
    *   Compute Power Spectral Density (PSD) using Welch's method.
    *   Estimate the **Noise Floor** (Average power at high frequencies).
    - **Validation:** Compare PSD corner frequency with derived $f_c$.
    *   Find the **Corner Frequency** $f_c$ where $PSD(f)$ intersects the Noise Floor.
3.  **Derivation:**
    *   Calculate Optimal Window $W_{opt} \approx 1 / (2 \pi f_c)$.
4.  **Verification:**
    *   Run Sig-RLS with this derived window.
    *   Compare MSE against the "Autotuned" or "Fixed 8s" result.

### PoC 6: Swing-Up CartPole (User Request)
- **Goal:** Control CartPole from downward position (`theta=pi`) to upright (`theta=0`).
- **Strategy:**
    - **Global Dynamics:** Use Sig-RLS (Degree 3) to learn non-linear physics globally.
    - **Control:** MPC (Random Shooting) with Energy-Shaping or Cosine Cost.
        - Cost: $J = (1 - \cos\theta) + 0.1 \dot{\theta}^2 + 0.01 x^2$.
    - **Control (Stabilization):** KRONIC LQR.
        - Learn Koopman Operator $A_z, B_z$ in Signature Space.
        - Map State Cost $Q_x \to Q_z$ via learned projection.
        - Solve LQR for gain $K$ acting on features.
    - **Workflow:** Reuse cached model from PoC 5, fine-tune if necessary.

## Hypothesis
The spectral analysis should derive a window size close to **8.0 seconds** (the empirically observed optimum from the previous benchmark), confirming the physical basis of the method.
