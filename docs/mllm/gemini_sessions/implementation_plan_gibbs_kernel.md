# Implementation Plan: Gibbs Kernel (State-Dependent Bandwidth)

## Theory (Rasmussen & Williams, Ch 4.2)
Stationary kernels (like RBF) assume the smoothness of the function is constant everywhere.
For CartPole swing-up:
-   **Equilibrium ($x \approx 0$):** High data density, highly non-linear control surface. Needs **short length scale** ($l \approx 0.5$) for precision.
-   **Swing-Up ($x$ far):** Sparse data, smooth physics. Needs **long length scale** ($l \approx 5.0$) for support/extrapolation.

The **Gibbs Kernel** is the principled non-stationary covariance function:
$$k(x, y) = \left( \frac{2 l(x) l(y)}{l(x)^2 + l(y)^2} \right)^{D/2} \exp \left( - \frac{\|x-y\|^2}{l(x)^2 + l(y)^2} \right)$$
where $l(x)$ is an arbitrary positive function of $x$.

## Implementation Strategy

### 1. `src/models/kernels.py`: Add `GibbsKernel`
-   **Input**: `l_func` (callable) or internally computed $l(x)$ using k-NN.
-   **Method**: `fit(X)` builds a KDTree to define $l(x)$ as the distance to the $k$-th neighbor.
-   **Computation**:
    -   Precompute $l_i$ for all centers $X$.
    -   Use broadcasting to compute the prefactor matrix $P_{ij} = \sqrt{\frac{2 l_i l_j}{l_i^2 + l_j^2}}$.
    -   Use broadcasting to compute the exponent denominator $D_{ij} = l_i^2 + l_j^2$.
    -   Compute weighted distance $E_{ij} = \|x_i - x_j\|^2 / D_{ij}$.
    -   $K_{ij} = P_{ij} \times \exp(-E_{ij})$.

### 2. `poc_rkhs_cartpole.py`: Integration
-   Replace `PeriodicKernel` with `GibbsKernel`.
-   Configure k-NN parameters (e.g., $k=10$ neighbors).
-   **Note on Periodicity**: Gibbs is natively RBF-like. To handle periodicity ($\theta$), we must embed $\theta \to (\cos \theta, \sin \theta)$ *before* passing to the kernel, OR modify Gibbs to use periodic distance.
    -   *Decision:* The current `get_features` already returns $[\cos, \sin]$. However, `PeriodicKernel` handled raw $\theta$.
    -   *Adaptation:* We will pass the *lifted* features (with cos/sin) to Gibbs, making it a standard RBF in lifted space. This simplifies things.

## Plan Steps
1.  Create `GibbsKernel` class.
2.  Update PoC to use `GibbsKernel`.
3.  Tune $k$ (neighbors) if needed (start with $k=10$).
4.  Run Verification (15s simulation).
