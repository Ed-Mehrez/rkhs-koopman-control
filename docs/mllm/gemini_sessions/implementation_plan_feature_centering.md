# Implementation Plan - Strict Feature Centering

## Problem
The learned Koopman model exhibits "drift" at the equilibrium point ($|A \phi(0)| \neq 0$). This causes the high-gain controller to fight phantom dynamics, destabilizing the system. Ad-hoc feedforward cancellation is undesirable.

## Solution: Strict Feature Centering
We will enforce the constraint $f(0)=0$ by shifting the feature map itself.

Define the centered feature map $\phi'(x)$:
$$\phi'(x) = \phi(x) - \phi(x_{eq})$$

Where $x_{eq} = [0, 0, 0, 0]$.

**Properties:**
1.  **Zero at Equilibrium:** $\phi'(x_{eq}) = \phi(x_{eq}) - \phi(x_{eq}) = \vec{0}$.
2.  **Zero Drift:** For any linear operator $A$, $A \phi'(x_{eq}) = A \vec{0} = \vec{0}$. (Assuming no intercept term in regression).
3.  **Generalizable:** Works for any kernel (RBF, Poly, etc.) and any system where the fixed point is known.

## Proposed Changes to `poc_rkhs_cartpole.py`

1.  **Calculate Reference Features:**
    - Compute `z_ref = kernel(target_state, centers)`.
    - Compute `state_ref` for the auxiliary state features.

2.  **Update `get_features` Function:**
    - Modify the function to subtract `z_ref` and `state_ref` from the raw features.
    - `return np.hstack([k_feat - z_ref, s_feat - state_ref])`.

3.  **Consistency:**
    - Apply this computed `z_ref` globally to all training data.
    - Ensure regression `fit_intercept=False` (default for standard least squares on centered data).

4.  **Verification:**
    - The `DRIFT CHECK` metric should drop to effectively **0.0** (machine precision).
    - The simulation should stabilize without the `u_ff` hack.

## Verification Plan
1.  **Run Simulation:** Execute `poc_rkhs_cartpole.py`.
2.  **Check Drift:** Verify printed drift is $\approx 0$.
3.  **Check Stability:** Verify pole remains upright.
