# RKHS-KRONIC Research Progress Summary

## What Has Been Proven to Work

| System | Task | Approach | Status |
|--------|------|----------|--------|
| Double Well | Saddle stabilization | Polynomial EDMD + DARE | ✓ Works |
| Double Well | Saddle stabilization | RBF Kernel SDRE | ✓ Works |
| CartPole | Local stabilization | Linear Koopman + DARE | ✓ Works perfectly |
| CartPole | Energy proxy discovery | RBF Kernel EDMD (λ≈1) | ✓ Correlates with physics |

## What Has Not Been Solved (Generalizable)

| System | Task | Approaches Tried | Result |
|--------|------|------------------|--------|
| CartPole | Swing-Up | Bilinear SDRE | ✗ Local tangent approx fails |
| CartPole | Swing-Up | Spectral Potential | ✗ Sign ambiguities, local minima |
| CartPole | Swing-Up | State Reconstruction | ✗ Vanishing gradients |
| General | Global control | Any finite-rank method | ✗ Cannot represent continuous spectrum |

## Theoretical Root Cause

**The Continuous Spectrum Problem:**

For Hamiltonian systems (like pendulum swing-up), the Koopman operator has *continuous spectrum* corresponding to:
- Periodic orbits (infinitely many energy levels)
- Heteroclinic connections (saddle manifolds)

**Implication**: No finite matrix $K \in \mathbb{R}^{n \times n}$ can exactly represent these dynamics. Kernel methods give us $N \times N$ Gram matrices (where N = data points), which is still finite.

## Key Discoveries

1. **Energy Proxy**: Kernel EDMD *can* identify an eigenfunction correlated with true energy (ρ ≈ 0.3-0.9 depending on kernel/data).

2. **Pumping Works**: The "energy control" law $u \propto (\phi_E - \phi_{target}) \cdot \nabla \phi_E \cdot g$ successfully *pumps* energy (pendulum spins wildly).

3. **Catching Fails**: Transitioning from "pump" to "stabilize" requires either:
   - Ad-hoc switching logic (not general)
   - A second eigenfunction with λ < 1 (hard to reliably identify)
   - MPC-style optimization (user rejects as "not KRONIC")

## Honest Assessment of Paths Forward

### Path A: Accept Hybrid Control
- Use Kernel LQR for local stabilization (works)
- Use heuristic for global phase (e.g., energy pumping)
- **Pro**: Practical, can work
- **Con**: Not a unified framework, requires problem-specific logic

### Path B: Infinite-Dimensional Control Theory
- Formulate control directly in RKHS as optimization problem
- Use functional gradient descent on cost $J[\phi]$
- **Pro**: Theoretically principled
- **Con**: Essentially becomes MPC in function space, computationally intensive

### Path C: Focus on Systems with Discrete Spectrum
- Apply RKHS-KRONIC to systems where it's theoretically sound:
  - Dissipative systems (attractors)
  - Stochastic systems (noise destroys continuous spectrum)
  - High-dimensional systems where local control suffices
- **Pro**: Theoretically justified success
- **Con**: Doesn't solve the "hard" problems

### Path D: Learning the Switching Function
- Learn a *classifier* for which regime (pump vs stabilize)
- Combine with regime-specific controllers
- **Pro**: Data-driven switching
- **Con**: Still hybrid, just with learned heuristics

## Recommended Next Step

Before more implementation, clarify:
1. Is the goal to solve swing-up *at all costs*, or to develop a *principled general framework*?
2. If principled framework: accept theoretical limitations for Hamiltonian systems
3. If practical swing-up: accept hybrid control with learned switching
