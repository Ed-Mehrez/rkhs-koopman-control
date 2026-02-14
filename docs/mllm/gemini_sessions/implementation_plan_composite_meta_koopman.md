# Implementation Plan: Composite Kernel & Meta-Koopman

## Overview

Two approaches to handle the continuous spectrum problem in a principled, data-driven way.

---

## Approach 1: Composite Kernel with Learned Gating

### Mathematical Formulation

$$k_{global}(x, y) = \alpha(x) \alpha(y) \cdot k_{pump}(x, y) + (1 - \alpha(x))(1 - \alpha(y)) \cdot k_{stab}(x, y)$$

where:
- $k_{pump}$ = kernel good for energy pumping (e.g., broad RBF for global dynamics)
- $k_{stab}$ = kernel good for stabilization (e.g., narrow RBF for local dynamics)
- $\alpha(x) \in [0, 1]$ = **learned gating function**

### Key Insight: $\alpha$ as a Koopman Eigenfunction

The gating function $\alpha(x)$ should naturally distinguish "far from target" vs "near target":
$$\alpha(x) = \sigma(\phi_E(x) - \phi_E(x^*))$$

where $\phi_E$ is the energy eigenfunction (discovered from Kernel EDMD) and $\sigma$ is a sigmoid.

This makes the switching **endogenous** - it emerges from the learned spectral geometry!

### Algorithm
```
1. Learn autonomous Kernel EDMD → Get eigenfunctions {φ_i}
2. Identify energy proxy: φ_E with λ ≈ 1, high variance
3. Define gating: α(x) = sigmoid(φ_E(x) - φ_E(target))
4. Build composite Gram matrix:
   G_global = α α^T ⊙ G_pump + (1-α)(1-α)^T ⊙ G_stab
5. Learn bilinear Koopman on G_global
6. Apply SDRE control using the composite representation
```

### Why This Might Work
- Near upright: α ≈ 0, controller uses $k_{stab}$ (local dynamics)
- Far from upright: α ≈ 1, controller uses $k_{pump}$ (global dynamics)
- Transition is smooth, governed by learned eigenfunction

---

## Approach 2: Meta-Koopman on Energy Manifold

### The Continuous Spectrum Problem

The Koopman operator for Hamiltonian systems has continuous spectrum because:
- Each energy level $E$ defines an invariant manifold $\mathcal{M}_E$
- On $\mathcal{M}_E$, dynamics are periodic (closed orbits)
- Different $E$ → different frequencies → continuous accumulation

### The Lift Idea

**Augment the state with energy**: $(x, E) \in \mathbb{R}^{n+1}$

On this extended space:
- Energy $E$ is (approximately) constant along trajectories
- The Koopman operator decomposes: $\mathcal{K} = \bigoplus_E \mathcal{K}_E$
- Each $\mathcal{K}_E$ has **discrete spectrum** (single frequency per orbit)

### Practical Implementation: "Koopman per Energy Band"

Discretize energy into bands $\{E_1, E_2, ..., E_K\}$:
```
1. Collect data with varying initial conditions
2. Compute energy for each sample: E_i = (1/2)θ̇² + (1 - cos θ)
3. Assign each sample to energy band k
4. Learn separate Koopman operators {K_1, ..., K_K}
5. At control time:
   a. Compute current energy E
   b. Interpolate between adjacent Koopman operators
   c. Apply appropriate control
```

### Kernel Formulation

Define **energy-conditioned kernel**:
$$k_E(x, y) = k(x, y) \cdot w(E(x), E(y))$$

where $w(E, E')$ is a weight that peaks when $E \approx E'$ (e.g., Gaussian).

This means:
- Points at similar energy are "close" in RKHS
- The Gram matrix naturally respects the energy foliation

### Control Strategy

After decomposition into energy bands:
1. **Pump phase**: Use $K_{low}$ to $K_{mid}$ to increase energy
2. **Stabilize phase**: Use $K_{high}$ (near target energy) for local control

The transition is automatic - we just use the Koopman corresponding to current energy.

---

## Proposed Experiment: Composite Kernel First

The composite kernel approach is simpler to implement and test. 

### File: `poc_composite_kernel_control.py`

1. Learn two kernels: RBF(σ=2.0) for pump, RBF(σ=0.5) for stab
2. Learn energy eigenfunction α
3. Build composite Gram matrix
4. Learn bilinear model
5. Control via SDRE
6. Test on CartPole swing-up

### Success Criteria
- Swings up from θ=π without hand-coded switching
- Stabilizes at θ=0 using the same codebase
- Generalizes to Double Well (crossing + stabilization)

---

## Theoretical Notes

### Connection to Mixture of Experts
The composite kernel is a **product of experts** variant:
$$k = \prod_i k_i^{w_i(x, y)}$$

This is known in GP literature (Hinton's Product of Experts).

### Connection to Koopman for Parametric Systems
The meta-Koopman idea relates to:
- **Parametric Koopman** (Kutz et al.): Koopman operators indexed by parameters
- **Extended DMD**: Augmenting state with slow variables
- **Floquet-Koopman**: Periodic systems have discrete Floquet exponents

### What Makes This "KRONIC-Style"
- No MPC optimization loop
- Control is a function in the RKHS
- Switching is via learned spectral geometry
- Infinite-dimensional via kernel trick
