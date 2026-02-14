
# Implementation Plan: Continuous Control (CARE) for CartPole

## Objective
Simplify the control strategy by moving to **Continuous Algebraic Riccati Equation (CARE)**. This avoids the ill-conditioning of Discrete ARE when $\Delta t \to 0$ and aligns with the user's intuition of "fast actuation".

## Proposed Changes
### `poc_rkhs_cartpole.py`
1.  **Kernel**: Retain `PeriodicKernel` (Fundamental for $S^1$ topology).
2.  **Regression**:
    *   Revert to Standard Linear Regression: $Z_{next} = A Z + B u$.
    *   Do NOT use Bilinear terms ($N$), simplifiying the learning problem.
    *   Collect data at native $dt=0.02s$ (No `lag_steps` or with minimal lag).
3.  **Control Synthesis**:
    *   Convert learned Discrete Operators $(A_d, B_d)$ to Continuous $(A_c, B_c)$:
        *   $A_c \approx \frac{A_d - I}{\Delta t}$ (or `scipy.linalg.logm`)
        *   $B_c \approx \frac{B_d}{\Delta t}$ (or `(A_d - I)^{-1} A_c B_d`)
    *   Solve CARE: `scipy.linalg.solve_continuous_are(Ac, Bc, Q, R)`.
    *   This provides a Gain $K$ valid for continuous feedback $u(t) = -K z(t)$.

## Verification
*   **Eigenvalues**: Check Real part of eigenvalues of $A_c$. (Unstable should be positive numbers).
*   **Simulation**: Run swing-up at 50Hz ($dt=0.02$).
