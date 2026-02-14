# Implementation Plan - Restoring Polynomial Kernel

## Goal
Prove that the **Polynomial Kernel** ($d=3$) can solve the Double Well control problem, matching the success of explicit **Monomial Features**.
The user correctly identified that "Concentric Circles" (failure) implies a bug or configuration error, as the bases are theoretically equivalent.

## Diagnosis
*   **Benchmark (`poc_klus_2d_control.py`)**: Used $N=15,000$ samples and explicit bilinear structure.
*   **Failure Case**: Used $N=3,000$ samples.
*   **Hypothesis**: Polynomials require high data density to "cancel out" higher-order terms ($u^2, u^3$) that are not present in the physics ($x' = f(x) + g(x)u$).

## Proposed Changes

### 1. Modify `poc_rkhs_poly_retry.py`
*   **Kernel**: Revert to `PolynomialKernel(degree=3, c=1.0)`.
*   **Data**: Increase $N_{train}$ to **15,000**.
    *   Uniform: 6,000
    *   Saddle Static: 4,000
    *   Saddle Bursts: 5,000 (Crucial for learning the unstable flow).
*   **Scaling**: Ensure `StandardScaler` is active (it is by default in `KernelEDMD`).

### 2. Verification
*   Run the script using `Agg` backend (headless).
*   Check for convergence ($\text{dist} < 0.1$).
*   Check Eigenfunction plot for correct separatrix (not circles).

## Verification Plan
### Automated Tests
*   Run `python examples/proof_of_concept/poc_rkhs_poly_retry.py`.
*   Expect `Converged` in stdout.

### Manual Verification
*   Inspect `poc_rkhs_poly_retry.png` (will create `double_well_poly_final.png`).
*   Confirm Eigenfunction 2 has split positive/negative regions across the y-axis (Left/Right wells).
