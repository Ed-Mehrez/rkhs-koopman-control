# Implementation Plan: RKHS-KRONIC (Kernel EDMD)

## References
*   **Klus et al. (2019)**: "Eigendecompositions of Transfer Operators in Reproducing Kernel Hilbert Spaces"
*   **Method**: Kernel EDMD (kEDMD) for the Koopman Operator.

## Core Philosophy
We move away from explicit feature maps (Polynomials, RFFs) and work directly in the **Reproducing Kernel Hilbert Space (RKHS)**.
The complexity of the algorithm will scale with the **Number of Samples ($N$)**, not the dimension of the system ($D$).
This allows us to handle high-dimensional systems (like images or large physics models) as long as we have a moderate number of samples.

## 1. Kernel EDMD (kEDMD) Formulation
We approximate the Koopman operator $\mathcal{K}$ using the empirical estimator derived in Klus et al. (Table 3):
*   **Eigenvalue Problem**:
    $$ (G_{XX} + n\varepsilon I)^{-1} G_{YX} \mathbf{v} = \lambda \mathbf{v} $$
    where $G_{XX} = k(X, X)$ and $G_{YX} = k(Y, X)$ are Gram matrices.
*   **Eigenfunctions**:
    The eigenfunction $\varphi_j(x)$ associated with eigenvector $\mathbf{v}_j$ is:
    $$ \varphi_j(x) = \mathbf{v}_j^\top G_{xX}(x) = \sum_{i=1}^n v_{ji} k(x_i, x) $$

## 2. KRONIC Control in RKHS
To perform control, we need the "Bilinear Structure" in the eigenfunction coordinates $z = [\varphi_1, \dots, \varphi_r]^\top$:
$$ \dot{z} = \Lambda z + \mathcal{B}_z(x) u $$
where $\mathcal{B}_z(x) \in \mathbb{R}^{r \times q}$ is the State-Dependent Control Matrix.

**The Novelty (Analytic Kernel Gradient)**:
We compute $\mathcal{B}_z(x)$ by analytically differentiating the eigenfunction expansion:
$$ \nabla_x \varphi_j(x) = \sum_{i=1}^n v_{ji} \nabla_x k(x_i, x) $$
Then:
$$ \mathcal{B}_z(x) = [ \nabla \varphi_1 \cdot B_{sys}, \dots, \nabla \varphi_r \cdot B_{sys} ]^\top $$
This allows us to use **finite-rank LQR/SDRE** on the low-dimensional intrinsic manifold ($r \approx 10-20$) while the state space can be arbitrarily complex.

## 3. Implementation Steps

### Step 1: `src/kedmd_core.py`
Implement the `KernelEDMD` class.
*   **Init**: Kernel type ('gaussian', 'polynomial'), bandwidth $\sigma$.
*   **Fit**: High-performance solver for the generalized eigenvalue problem.
    *   *Optimization*: Use Cholesky decomposition or truncated SVD for stability if $N$ is large.
*   **Predict**: Evaluate $\varphi(x)$.
*   **Jacobian**: Evaluate $\nabla \varphi(x)$.

### Step 2: `models/kernels.py`
Implement Kernels and their gradients with **Stability Protections**.
*   **Gaussian RBF**: $k(x, y) = \exp(-\|x-y\|^2 / 2\sigma^2)$.
    *   **Instability Risk**: If $\sigma$ is small, gradients explode/vanish.
    *   **Fix**: Implement **Median Heuristic** for automatic $\sigma$ selection: $\sigma^2 = \text{median}(\|x_i - x_j\|^2)$.
    *   **Analytic Gradient**: $\nabla_x k(x, y) = -\frac{1}{\sigma^2} k(x, y) (x - y)$.
*   **Polynomial**: $k(x, y) = (x^\top y + c)^d$.
*   **Matern**: Optional, for rougher dynamics.

### Step 3: `examples/poc_rkhs_double_well.py` (Validation)
Re-run the Anisotropic Double Well experiment using `KernelEDMD` instead of `KoopmanLearner` (Poly).
*   Verify convergence to origin.
*   Compare sample efficiency vs Polynomials.

### Step 4: `examples/poc_rkhs_cartpole.py` (Scale-Up)
Apply to CartPole Swing-Up.
*   Collect interaction data (random actions).
*   Train kEDMD.
*   Swing up using KRONIC SDRE.

## User Review Required
*   Confirm choice of **Gaussian RBF** as the primary kernel.
*   Confirm we assume $B_{sys}$ (physics actuation matrix) is known for the gradient calculation. (If not, we'd need to learn a separate input kernel, which is harder).
