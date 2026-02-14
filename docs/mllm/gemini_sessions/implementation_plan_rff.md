# Implementation Plan - PoC 8.5: RBF Kernel via Random Fourier Features

## User Review Required
> [!NOTE]
> We are replacing Signatures (Polynomials) with **Random Fourier Features (RFF)**.
> **Why:** RFFs approximate the Gaussian (RBF) kernel: $k(x,y) = e^{-\gamma \|x-y\|^2} \approx \langle \phi(x), \phi(y) \rangle$.
> **Benefit:** The feature map $\phi(x) = [\cos(\omega_1 x + b_1), \dots]$ naturally captures **periodicity** and global topology, addressing the root cause of the Log-Sig swing-up failure.
> **Risk:** "Gradient Instability". If the length scale is too small (high $\omega$), the functions are "spiky" and derivatives are large, leading to noisy control. We must obtain a standard length scale (e.g., $\gamma=1.0$).

## Proposed Changes

### Reference Implementation (`poc_cartpole_control.py`)
#### [MODIFY] [poc_cartpole_control.py](file:///home/ed/SynologyDrive/Documents/Research/P&E_Research/rkhs_kronic/examples/proof_of_concept/poc_cartpole_control.py)
- **SigRLS_Dynamics**:
    - Remove `iisignature` dependency.
    - Add `sklearn.kernel_approximation.RBFSampler`.
    - Initialize `RBFSampler` with `gamma=1.0` (tuneable) and `n_components=400` (similar size to LogSig).
    - Update `update`, `predict`, `fit_koopman` to use `sampler.fit_transform`.

## Verification Plan
1.  **Swing-Up Test:** The primary metric. Can the global linear model on RFFs pump energy?
2.  **Stabilization:** Ensure we don't lose the local stability (RBFs should handle local smooth dynamics fine).
