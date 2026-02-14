# Implementation Plan: Sparse Kernel EDMD (Nyström Approximation)

## Problem
The `CartPole` (and `Double Well`) verification showed that Exact Kernel EDMD ($O(N^3)$) is too slow for $N > 5000$.
To reliably stabilize chaotic/unstable systems, we need high data density ($N \approx 10,000+$).
The current implementation solves the dual system on the full dataset.

## Solution: Sparse Approximation (Nyström)
Instead of using all $N$ points as kernel centers, we select a subset of $M \ll N$ centers (e.g., $M=2000$).
We use the full dataset $N$ to regress the weights on these $M$ centers.

### Math
*   Centers: $C = \{c_1, ..., c_M\} \subset X$.
*   Feature Map: $\Phi(x) = [k(x, c_1), ..., k(x, c_M)]^T$.
*   Problem: Find $A_{approx}$ such that $\Phi(x_{t+1}) \approx A \Phi(x_t)$.
*   Regression: Minimize $\sum || \Phi(y_i) - A \Phi(x_i) ||^2$.
*   This is solved via Primal Ridge Regression on the feature map $\Phi(x)$.
*   Cost: $O(N M^2 + M^3)$. For $N=10k, M=2k$, this is $\sim 100\times$ faster than $O(N^3)$.

## Changes

### 1. `src/kedmd_core.py`
*   Modify `fit(X, Y, n_centers=None)`:
    *   If `n_centers` is provided and $< N$:
        *   Select `X_centers` using K-Means (better) or Random Choice (faster). Let's use Random for now.
        *   Construct Feature Matrices:
            *   $Z = K(X, X_{centers})$ shape $(N, M)$.
            *   $Z_{next} = K(Y, X_{centers})$ shape $(N, M)$.
        *   Solve Primal: $A = (Z^T Z + \lambda I)^{-1} Z^T Z_{next}$ ($M \times M$).
        *   Eigen-decompose $A$ ($M \times M$).
        *   Store `coefs_` as eigenvectors of $A$. `eigenvectors` are vectors in $\mathbb{R}^M$.
    *   Update `predict_eigen`:
        *   If Sparse, `K_val = kernel(X_eval, X_centers)`.
        *   Result `K_val @ eigenvectors`.
    *   Update `get_control_matrix_B_phi`:
        *   Use gradients w.r.t centers.

### 2. `poc_rkhs_cartpole.py`
*   Update `kedmd.fit` call to use `n_centers=2000`.
*   Increase `N_train` to 10,000 (Dense coverage).

## Verification
*   Execute `poc_rkhs_cartpole.py`.
*   Expect rapid training (< 30s) even with 10k samples.
*   Verify Swing-Up success.
