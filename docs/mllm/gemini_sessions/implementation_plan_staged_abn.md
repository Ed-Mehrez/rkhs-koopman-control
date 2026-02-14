# Implementation Plan: Staged A/B/N Learning for CartPole

## Goal

Implement a principled staged learning approach for the bilinear control model:
```
dφ/dt = A·φ + B·u + N·(φ⊗u)
```

Instead of jointly regressing A, B, N (which conflates learned quantities), we decompose:
1. **A**: Learn from autonomous data (u=0) using KernelGEDMD
2. **B**: Learn from equilibrium-state data (x≈0) with varying u
3. **N**: Learn from residual using full controlled data

## Proposed Changes

### [MODIFY] [poc_rkhs_cartpole.py](file:///home/ed/SynologyDrive/Documents/Research/P&E_Research/rkhs_kronic/examples/proof_of_concept/poc_rkhs_cartpole.py)

#### Data Collection (Lines ~90-130)
Replace current random data collection with three-stage collection:

```python
# Stage 1: Autonomous data (u=0)
X_auto, X_dot_auto = collect_autonomous_data(env, n_samples=3000)

# Stage 2: Equilibrium data (x≈0, varying u)
X_eq, U_eq, X_dot_eq = collect_equilibrium_data(env, n_samples=2000)

# Stage 3: Full controlled data (random x, random u)
X_ctrl, U_ctrl, X_dot_ctrl = collect_controlled_data(env, n_samples=5000)
```

#### Learning Stage 1: A from Autonomous (New Section ~200)
```python
# Use KernelGEDMD on autonomous data
from src.kgedmd_core import KernelGEDMD, RBFKernel
gedmd = KernelGEDMD(kernel=RBFKernel(sigma=1.0))
gedmd.fit(X_auto[:, 1:], X_dot_auto[:, 1:])  # Exclude cart position

# Extract Koopman eigenvalues as A diagonal
# φ̇ = λφ  →  A[i,i] = λ_i
A_cont = np.diag(gedmd.eigenvalues_[:n_eigs].real)
```

#### Learning Stage 2: B from Equilibrium (New Section ~230)
```python
# At x≈0: φ̇ = A·φ(0) + B·u ≈ B·u  (since φ(0) ≈ constant)
# Regress: φ̇_eq = B @ U_eq.T

Z_eq = get_features(X_eq)
Z_dot_eq = compute_feature_derivatives(X_eq, X_dot_eq)

# Remove A term (φ̇ - A·φ = B·u)
residual_eq = Z_dot_eq - Z_eq @ A_cont.T
B_cont = np.linalg.lstsq(U_eq, residual_eq, rcond=None)[0].T
```

#### Learning Stage 3: N from Residual (New Section ~260)
```python
# Full data: φ̇ = A·φ + B·u + N·(φ⊗u)
# Residual: r = φ̇ - A·φ - B·u = N·(φ⊗u)

Z_ctrl = get_features(X_ctrl)
Z_dot_ctrl = compute_feature_derivatives(X_ctrl, X_dot_ctrl)

residual_ctrl = Z_dot_ctrl - Z_ctrl @ A_cont.T - U_ctrl @ B_cont.T
Z_times_U = Z_ctrl * U_ctrl  # Bilinear term

# Regress: residual = N @ (Z * U)
N_cont = np.linalg.lstsq(Z_times_U, residual_ctrl, rcond=None)[0].T
```

#### Validation (New Section ~280)
Add R² computation for each stage:
```python
# Validate A (autonomous prediction)
Z_dot_pred_auto = Z_auto @ A_cont.T
R2_A = 1 - np.sum((Z_dot_auto - Z_dot_pred_auto)**2) / np.sum((Z_dot_auto - Z_dot_auto.mean(0))**2)

# Validate A+B (equilibrium prediction)
# Validate A+B+N (full model prediction)
```

## Verification Plan

### Automated Tests
```bash
# Run the modified poc_rkhs_cartpole.py
cd /home/ed/SynologyDrive/Documents/Research/P&E_Research/rkhs_kronic/
/home/ed/miniconda3/envs/rkhs-kronic-gpu/bin/python -u examples/proof_of_concept/poc_rkhs_cartpole.py 2>&1 | head -100
```

**Expected Output:**
- Stage 1 R² (A only) > 0.7 on autonomous data
- Stage 2 R² (A+B) > 0.8 on equilibrium data  
- Stage 3 R² (A+B+N) > 0.9 on full controlled data
- CARE stabilizable at target state
- Control simulation shows θ → 0° (stabilization)

### Manual Verification
1. Check generated plot shows pole angle converging toward 0°
2. Check GIF animation shows physical stabilization behavior
