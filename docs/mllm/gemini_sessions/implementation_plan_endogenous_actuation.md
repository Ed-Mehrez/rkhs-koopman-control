# Implementation Plan: Endogenous Actuation Learning

## Goal

Make the Spectral Potential Controller **truly general** by removing hardcoded physics (the B matrix). Instead, learn the control influence $B_{spectral}$ directly from data using Input-Kernel EDMD.

## Core Logic: Input-Kernel EDMD

1.  **Data**: Collect triples $(x_k, u_k, x_{k+1})$.
2.  **Fitting**:
    Solve the regression problem in feature space:
    $$ \Psi(x_{k+1}) \approx \mathcal{K} \Psi(x_k) + \mathcal{B} u_k $$
    where $\mathcal{K}$ is the Koopman operator and $\mathcal{B}$ is the Input operator in the kernel space.
    
    This is solved via Ridge Regression on the augmented matrix $[\Psi(X), U]$.

3.  **Eigenfunction Sensitivity**:
    The eigenfunctions are $\phi(x) = v^T \Psi(x)$.
    The evolution is $\phi(x_{k+1}) = \lambda \phi(x_k) + (v^T \mathcal{B}) u_k$.
    Therefore, the sensitivity of the *next* eigenfunction value to the *current* control is:
    $$ \frac{\partial \phi_{next}}{\partial u} = v^T \mathcal{B} $$
    This vector (size $1 \times dim\_u$) tells us exactly how to push $u$ to change $\phi$.

4.  **Control Law**:
    Minimize potential $V(\phi_{next}) = (\phi_{next} - \phi_{tgt})^2$.
    $$ \nabla_u V = 2(\phi_{next} - \phi_{tgt}) \frac{\partial \phi_{next}}{\partial u} $$
    $$ u = -k \cdot \text{sign}(\nabla_u V) $$

## Proposed Changes to `general_spectral_potential.py`

### `SpectralController.fit`
- Accept `U` data.
- Construct `G_aug = [Psi(X), U]`.
- Solve for `[K_T; B_T]`.
- Store `self.B_psi = B_T.T`.

### `SpectralController.get_control`
- Remove `B_matrix` argument.
- Compute `dphi_E_du = v_E @ self.B_psi`.
- Compute `dphi_S_du = v_S @ self.B_psi`.
- Compute `grad_u` using these learned sensitivities.

## Verification
- Run on **Double Well**. (Should identify $\frac{d\phi}{du}$ correctly and cross saddle).
- Run on **CartPole**. (Should identify complex relationship between $u$ and energy mode).
- No problem-specific code remains in the controller.
