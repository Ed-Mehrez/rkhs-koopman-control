
# Implementation Plan - Action Repeat for RKHS CartPole

## Goal
Improve Signal-to-Noise ratio for "Control B Matrix" learning by using Action Repeat (`lag_steps > 1`). Since changes in 0.02s are tiny, the Kernel EDMD struggles to distinguish Control drift from State drift.

## Proposed Changes
1.  **Modify Data Collection**:
    *   Define `lag_steps = 4`.
    *   Instead of `env.step(u)`, run a loop of `lag_steps` applying `u`.
    *   Store `X` and `Y` (where `Y` is state after k steps).
2.  **Modify Simulation Loop**:
    *   In the control loop, after computing `u`, apply it for `lag_steps` in the environment.
    *   This aligns the physics with the trained discrete map $K^t$.

## Verification
*   Check `|B|` stats printed during training (should increase by ~4x).
*   Check if controller successfully swings up.
