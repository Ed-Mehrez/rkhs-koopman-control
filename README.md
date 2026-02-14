# RKHS Koopman Control

Data-driven control via Koopman operators and kernel methods.

## Key Results

| System | Method | Result |
|--------|--------|--------|
| **CartPole** | Local Koopman SDRE | **100% success rate** |
| **Kuramoto 20-D** | Sparse k-NN | **Stable for 10k steps** |
| **Cost Reconstruction** | Kernel inverse | **0.12% mean error** |
| **Speedup** | k-NN vs full | **88x faster** |

## Core Algorithm: Kernel GEDMD

Kernel Extended Dynamic Mode Decomposition following Klus et al.:

```python
# Koopman transfer operator
K_E = C_XY @ inv(C_XX + εI)

# Where:
# C_XX = Φ(X)ᵀ Φ(X)  (Gram matrix)
# C_XY = Φ(X)ᵀ Φ(Y)  (cross-covariance)
# Φ(x) = kernel feature map
```

### SDRE Control in Eigenspace

State-Dependent Riccati Equation (SDRE) solved in Koopman eigenspace:
1. Project state to eigenfunctions: z = φ(x)
2. Solve SDRE: A(z)ᵀP + PA(z) - PBR⁻¹BᵀP + Q = 0
3. Compute gain: K = R⁻¹BᵀP
4. Apply control: u = -Kz

## Quick Start

```bash
# Setup
conda env create -f environment.yml
conda activate rkhs-koopman-control

# Run CartPole stabilization
python examples/experiment_local_koopman_sdre.py
```

## Usage

```python
from src.kronic_controller import KRONICController
from src.kgedmd_core import KernelGEDMD, RBFKernel

# Learn Koopman operator from data
kernel = RBFKernel(sigma=1.5)
kgedmd = KernelGEDMD(kernel, regularization=1e-5)
kgedmd.fit(X, X_next)

# Create controller
controller = KRONICController(kgedmd, Q=np.eye(n), R=0.1*np.eye(m))

# Control loop
for t in range(horizon):
    u = controller.compute_control(x)
    x = env.step(u)
```

## Project Structure

```
rkhs-koopman-control/
├── src/
│   ├── kronic_controller.py    # Main KRONIC controller
│   └── kgedmd_core.py          # Kernel EDMD implementation
├── examples/
│   ├── experiment_local_koopman_sdre.py   # CartPole (100% success)
│   ├── experiment_kuramoto_scaling.py     # 20-D scaling
│   └── proof_of_concept/
│       ├── poc_rkhs_cartpole*.py          # CartPole variants
│       └── poc_double_well*.py            # Double-well control
└── docs/
    ├── convex_kernel_control_theory.md
    ├── sdre_eigenspace_control_theory.md
    └── principled_rkhs_sdre_theory.md
```

## Key Innovations

### 1. Local Koopman SDRE
Sparse k-NN selection (200/5000 samples) for 88x speedup without accuracy loss.

### 2. Kernel Cost Reconstruction
Learn cost function from expert demonstrations:
```
J(x) = ||K(x, ·)||² in RKHS
```
Achieves 0.12% mean error across 6 benchmark systems.

### 3. Bilinear SDRE for Control-Affine Systems
Extended formulation for systems with control-affine structure:
```
dx/dt = A(x)x + B(x)u
```

## Theoretical Foundation

### Koopman Operator Theory
For a dynamical system dx/dt = f(x), the Koopman operator K acts on observables:
```
(Kg)(x) = g(f(x))
```

Eigenfunctions satisfy: Kφ = λφ

### RKHS Embedding
Kernel methods provide finite-dimensional approximation:
- RBF kernel: k(x,y) = exp(-||x-y||²/2σ²)
- Eigenfunction approximation via Nyström method

See [docs/](docs/) for complete theory.

## Benchmarks

### CartPole Swing-Up
- Method: Local Koopman SDRE with sparse k-NN
- Success: 100% (50/50 trials)
- Time: 0.8ms per control computation

### Kuramoto Oscillator (20-D)
- Method: Full Kernel GEDMD
- Stability: 10,000+ steps without divergence
- Synchronization achieved in ~500 steps

## Citation

```bibtex
@article{rkhs-koopman-control,
  title={Data-Driven Control via Koopman Operators in RKHS},
  author={Mehrez, Edward},
  journal={In preparation for IEEE TAC},
  year={2026}
}
```

## References

- Klus et al. "Data-Driven Model Reduction and Transfer Operator Approximation"
- Williams et al. "A Data-Driven Approximation of the Koopman Operator"
- Kaiser et al. "Data-Driven Discovery of Koopman Eigenfunctions for Control"

## License

MIT License

---
*Split from [RKHS-KRONIC](https://github.com/Ed-Mehrez/RKHS-KRONIC) repository.*
