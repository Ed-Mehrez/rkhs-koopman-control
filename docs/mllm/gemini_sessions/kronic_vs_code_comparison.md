# Comparison: KRONIC Theory vs. Code vs. Our Work

## 1. Theoretical Framework (KRONIC.pdf)
The paper **"Data-driven discovery of Koopman eigenfunctions for control" (Kaiser et al.)** explicitly derives the bilinear control structure.

*   **Equation 31**: Defines the dynamics in eigenfunction coordinates:
    $$ \frac{d}{dt}\phi(x) = \Lambda \phi(x) + \nabla_x \phi(x) \cdot B u $$
    This is inherently **Bilinear** because $\nabla \phi(x)$ depends on $x$.
*   **equation 33**: Explicitly formulates the **State-Dependent Riccati Equation (SDRE)**:
    $$ Q + P\Lambda + \Lambda^T P - P B_\phi(x) R^{-1} B_\phi^T(x) P = 0 $$
    where $B_\phi(x) = \nabla \phi \cdot B$.

**Verdict**: The theory *fully supports* and anticipates the approach we discovered.

## 2. Prior State of THIS Repository (`src/kronic_controller.py`)
The existing code in *this specific workspace* appears to be a **Simplified Replication** of the KRONIC framework, likely created by previous researchers trying to reproduce specific examples.
*   **Limitation**: It only implemented the standard Linear LQR (`control.lqr`) with constant matrices.
*   **Gap**: It did not yet include the advanced Bilinear/SDRE features described in the original paper's Eq. 33.

## 3. Our New Implementation (`poc_klus_2d_control.py`)
We have effectively **advanced the replication** of the KRONIC paper within this repository.
1.  **Bilinear Model**: We learned $z_{next} = A z + B u + \sum u_i N_i z$, which captures the term $\nabla \phi \cdot B$ described in the paper.
2.  **SDRE Solver**: We implemented the solver for Eq 33 (Paper Sec 4.4), adapting the gain $K(x)$ locally.
3.  **Result**: We have successfully replicated the *advanced* control theoretical claims of Kaiser et al., which were previously missing from this codebase.

## Conclusion
*   **Is it Novel Science?** No, the math and method are fully described in the original **Kaiser et al. (2021)** paper.
*   **Is it Novel for this Repo?** **YES.** We have upgraded the local codebase from a "Linear Replication" to a "Bilinear/SDRE Replication," matching the paper's full capability.
