# Implementation Plan: Unified Discrete EDMD + DARE for CartPole

## Goal

Implement a **principled, unified approach** for CartPole control using the discrete EDMD + DARE framework that worked on the double well. This validates the theoretical analysis (MLE justification, DARE robustness) and provides a stepping stone toward high-dimensional nonlinear control.

## Background

From our theoretical analysis:
1. **Discrete EDMD + DARE** is more robust than continuous kGEDMD + CARE
2. **Polynomial features** with proper state representation work well
3. **Bilinear structure** captures control-affine dynamics
4. The approach is **MLE-justified** under Gaussian noise assumptions

---

## Proposed Changes

### [NEW] `poc_discrete_cartpole.py`

Create a clean implementation following the validated `poc_klus_2d_control.py` structure:

#### 1. State Representation
```python
# CartPole raw state: [x, x_dot, theta, theta_dot]
# Lifted state for periodicity: [x, x_dot, cos(theta), sin(theta), theta_dot]
```

This handles the periodic nature of θ without kernel complications.

#### 2. Polynomial Features
```python
poly = PolynomialFeatures(degree=3, include_bias=True)
# On 5D lifted state → ~56 features
```

Polynomial features capture local dynamics and have the affine property needed for bilinear control.

#### 3. Bilinear Koopman Learning
```python
# Z_{k+1} = A Z_k + B u_k + N u_k Z_k (bilinear)
G = [Z, U, Z * U]
K = solve(G'G + λI, G'Z_next)
```

Same structure as double well.

#### 4. Control via DARE
```python
# DARE: P = A'PA - A'PB(R + B'PB)^{-1}B'PA + Q
P = scipy.linalg.solve_discrete_are(A, B(z), Q, R)
K = (R + B'PB)^{-1} B'PA
```

Using SDRE (State-Dependent Riccati Equation) for the bilinear part.

#### 5. Two-Phase Control
- **Phase 1: Stabilization** - SDRE controller when near upright
- **Phase 2: Swing-up** - Energy-based controller to pump energy into the system

---

## Verification Plan

### Automated Tests

1. **Run the new PoC**:
```bash
python examples/proof_of_concept/poc_discrete_cartpole.py
```
**Expected**: Stabilization works (starting from small θ), swing-up may require hybrid approach.

2. **Compare with double well baseline**:
```bash
python examples/proof_of_concept/poc_klus_2d_control.py
```
**Expected**: Output shows variance reduction with control frequency (already validated).

### Manual Verification

1. **Check eigenvalue spectrum** of learned discrete A matrix
   - Should have |\lambda| ≤ 1 for stable modes
   - The unstable upright mode should have |\lambda| slightly > 1

2. **Verify control performance**:
   - Stabilization from θ ≈ ±10° should converge to upright
   - Plot trajectories and control effort

---

## Key Design Decisions

1. **Polynomial over kernel features**: Polynomial basis has the linear-in-features property needed for bilinear control. Kernels introduce nonlinear feature evolution that breaks the z_{k+1} = Az + Bu structure.

2. **Trigonometric lifting for periodicity**: Rather than using periodic kernels, we lift [θ] → [cos(θ), sin(θ)]. This is a universal approach for periodic states.

3. **DARE over CARE**: Discrete formulation is MLE-justified and doesn't require derivative estimation.

4. **Bilinear over linear**: Captures the control-affine structure of mechanical systems.

---

## Success Criteria

- [ ] Stabilization from ±20° works reliably
- [ ] Learned A matrix has correct stability structure (|λ| ≈ 1 for upright mode)
- [ ] DARE converges without numerical issues
- [ ] Code is clean and reusable as template for other systems
