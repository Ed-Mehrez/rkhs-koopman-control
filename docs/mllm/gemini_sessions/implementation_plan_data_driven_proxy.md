# Implementation Plan: Data-Driven Energy Proxy Control

## Goal

Implement a **fully data-driven** CartPole swing-up and stabilization controller that does **not** assume knowledge of the physical energy function. Instead, it discovers an "energy proxy" eigenfunction from data and uses it for control.

## Theoretical Basis

1. **Hamiltonian Systems**: Have a conserved quantity (Energy) $H(x)$ such that $\dot{H} = 0$ (unforced).
2. **Koopman Spectrum**: The Koopman operator for such systems has an eigenvalue $\lambda = 1$ (continuous time $\mu=0$) with eigenfunction $\phi(x) = H(x)$.
3. **Control Strategy**: 
   - **Swing-up**: Drive the system to the energy level of the unstable equilibrium. Control law: $u = k \cdot \text{sign}(\dot{\theta} \cos\theta) \cdot (E - E_{target})$.
   - **Data-Driven**: Replace $E$ with discovered $\phi(x)$. Control law: $u = k \cdot \text{sign}(\hat{\dot{\phi}}) \cdot (\phi - \phi_{target})$.

## Implementation Steps

### [NEW] `poc_data_driven_cartpole.py`

#### 1. Unsupervised Learning Phase
- Collect **free-swing trajectories** (random initial states, zero or random control).
- Fit **Quadratic Koopman Model** (degree 2 polynomial features).
- **Identify Energy Proxy**:
  - Find eigenvector $v$ with eigenvalue $\lambda \approx 1$.
  - Selecting the *right* one: The one with highest variance or correlation with quadratic state terms.
  - Compute $\phi(x) = v^\top \Psi(x)$.

#### 2. Calibration
- Determine $\phi_{target}$: The value of $\phi$ at the upright state $x_{up} = [0,0,0,0]$.
- Determine 'pumping direction': Check sign of $\nabla \phi \cdot g_{control}$ to know which way to push.

#### 3. Control Loop
- **Swing-Up**: 
  - Estimate $\dot{\phi} \approx (\phi_t - \phi_{t-1})/\Delta t$.
  - $u = -K_{pump} \cdot (\phi - \phi_{target}) \cdot \text{sign}(\dot{\phi})$.
- **Switching**:
  - When $|\phi - \phi_{target}| < \epsilon$ and near upright state, switch.
- **Stabilization**:
  - Use Linear EDMD + DARE (already proven to work).

## Verification Plan

### Automated Tests
1. **Run `poc_data_driven_cartpole.py`**:
```bash
python examples/proof_of_concept/poc_data_driven_cartpole.py
```
2. **Metrics**:
   - Correlation of $\phi$ with true Energy (expect > 0.9).
   - Success rate of swing-up (reaching upright).
   - Stabilization accuracy.

### Manual Verification
1. **Visual Inspection**:
   - Plot $\phi(x)$ vs True Energy.
   - Plot trajectory $\theta(t)$ showing swing-up.

## Key Challenges & Mitigations
- **Multiple Conserved Quantities**: There might be other invariants (e.g. momentum if no friction). 
  - *Mitigation*: The energy proxy should be dominated by potential/kinetic terms (x_2, x_3 dependence).
- **Sign Ambiguity**: Eigenfunctions are unique up to scalar multiple.
  - *Mitigation*: Calibrate sign so that 'upright' is a maximum (or minimum).

---

## Success Criteria
- [ ] Discovery of $\phi(x)$ with >0.9 correlation to true Energy.
- [ ] Swing-up achieved using *only* $\phi(x)$ feedback.
- [ ] Smooth transition to DARE stabilization.
