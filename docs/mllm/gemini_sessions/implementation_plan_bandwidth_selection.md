# Implementation Plan - Variable Bandwidth Kernel

## Goal
Implement a Non-Stationary Kernel (Variable Bandwidth) to address the "Stability vs Support" trade-off.
-   **Equilibrium (Dense Data):** Needs sharp, high-bandwidth control. $\sigma_i$ should be small.
-   **Swing-Up (Sparse Data):** Needs broad, smooth support. $\sigma_i$ should be large.

This aligns with **Rasmussen & Williams (Ch 4.2)** on non-stationary covariance functions and is implemented via **k-NN density estimation**.

## Proposed Changes

### 1. `src/models/kernels.py`

#### [NEW] Class `VariableBandwidthRBF`
-   **Init:** Accepts `k` (neighbors) and `base_kernel` type (e.g. Gaussian or Periodic).
-   **Fit(X):**
    -   Build `KDTree` on `X` (centers).
    -   Query `k`-th nearest neighbor distance for each point $x_i$.
    -   Store `sigmas` vector.
-   **Call(X, Y):**
    -   Implement the "Basis Function" form (asymmetric):
        $$k(x_i, y) = \exp\left( -\frac{\text{dist}(x_i, y)^2}{2 \sigma_i^2} \right)$$
    -   *Note:* Strictly speaking, the symmetric Gibbs form is better for GPs, but for Feature Regression (RBF Network), the asymmetric form is standard and computationally simpler.
-   **Gradient:**
    -   Update gradient formula to use per-center `sigma[i]`.

### 2. `examples/proof_of_concept/poc_rkhs_cartpole.py`

#### [MODIFY] Kernel Instantiation
-   Replace `PeriodicKernel` with `VariableBandwidthRBF` (wrapping the periodic logic or subclassing).
-   Since we need Periodicity + Variable Bandwidth, we might need to modify `PeriodicKernel` to support `sigma` arrays.

**Refined Strategy for `kernels.py`:**
Modify `PeriodicKernel` to accept `use_variable_width=True`.
-   In `fit`: Compute k-NN distances -> `self.sigmas`.
-   In `__call__`: Use broadcasting to divide by `self.sigmas`.

## Detailed Logic (PeriodicKernel)

```python
    def fit(self, X):
        if self.use_variable_width:
            from scipy.spatial import cKDTree
            tree = cKDTree(X)
            # Query k=10
            dists, _ = tree.query(X, k=10)
            # Use distance to 10th neighbor as sigma
            self.sigmas = dists[:, -1]
            # Clip to avoid zero or extreme values
            self.sigmas = np.clip(self.sigmas, 0.1, 10.0)
        
    def __call__(self, X, Y=None):
        # ... compute dist_sq as before ...
        
        if self.use_variable_width:
             # If Y is None (Gram matrix), X is centers.
             # K[i, j] matches center X[i].
             # We want sigma_i associated with row i?
             # Actually, for features: Phi(y) = [k(x1, y), k(x2, y)...].
             # So x_i are the centers.
             # k(x_i, y) should use sigma_i.
             # X are centers?
             # In kgedmd_core, X_train is passed as X?
             pass 
```

## Verification Plan
-   **Run Simulation:** 15s horizon.
-   **Expectation:** Tight control at 0 (due to dense equilibrium data $\to$ small sigma). Robust recovery from perturbations (broad sigma elsewhere).
