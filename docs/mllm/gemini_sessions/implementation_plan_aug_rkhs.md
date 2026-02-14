
# Implementation Plan: State-Augmented RKHS Control

## Problem
Standard Kernel LQR failed ("Runaway Cart") because Kernel Features decay to zero far from the training data, leading to a "Zero Cost" trap for the LQR controller. The user rejected MPPI as it abandons the Operator-theoretic approach.

## Solution: Augmented EDMD
We will augment the dictionary of observables:
$$ z = \begin{bmatrix} \phi_{kernel}(x) \\ x_{state} \end{bmatrix} $$
This is a standard technique (e.g., Williams et al. EDMD) to ensure the state itself is always in the span of the dictionary.

### Advantages
1.  **Global Observability**: Even if $\phi(x) \to 0$ far away, $x$ remains.
2.  **Valid LQR Cost**: We can define $Q$ to penalize $x^T Q_x x$ directly ensuring $J \to \infty$ as $x \to \infty$.
3.  **Operator Theoretic**: We are still learning the Koopman Operator on this augmented space.

## Changes to `poc_rkhs_cartpole.py`
1.  **Feature Map**: return `hstack([kernel_features, x])`.
2.  **Regression**: Learn $A, B, N$ on this $(r+4)$ dimensional space.
3.  **CARE**: Define $Q$ matrix to penalize the last 4 indices (State) heavily.
4.  **Simulation**: Use standard Full-State Feedback $u = -K (z - z_{tgt})$.

## Verification
*   **Runaway Check**: $x$ should remain bounded (e.g. within $\pm 2.4$).
*   **Swing-Up**: Should pumping energy and catch at top.
