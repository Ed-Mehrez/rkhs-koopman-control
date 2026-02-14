# AGENTS.md

Context for AI coding agents working on this repository.

## Environment

**Use the shared conda environment** - do NOT create a new one:

```bash
conda activate rkhs-kronic-gpu
```

The `environment.yml` in this repo is for CI/collaborators only.

## Project Overview

Data-driven control via Koopman operators and kernel methods.

**Core Algorithm**: Kernel GEDMD (following Klus et al.)
```
K_E = C_XY @ inv(C_XX + εI)
```

## Key Files

| File | Purpose |
|------|---------|
| `src/kronic_controller.py` | Main KRONIC controller with SDRE |
| `src/kgedmd_core.py` | Kernel EDMD implementation |
| `examples/experiment_local_koopman_sdre.py` | CartPole (100% success) |
| `examples/experiment_kuramoto_scaling.py` | 20-D scaling test |

## Build & Test

```bash
# Run CartPole stabilization
python examples/experiment_local_koopman_sdre.py

# Run Kuramoto scaling
python examples/experiment_kuramoto_scaling.py
```

## Critical Knowledge

### Key Results
- CartPole: 100% success rate
- 88x speedup via sparse k-NN (200/5000 samples)
- 20-D Kuramoto: Stable for 10k+ steps
- Kernel cost reconstruction: 0.12% mean error

### Local Koopman SDRE
Use sparse k-NN selection for massive speedup without accuracy loss.

### RBF Kernel Parameters
Default: `sigma=1.5`, `regularization=1e-5`
These are universal across many systems.

## Conventions

- Python 3.10+
- NumPy/SciPy for numerics
- `gymnasium` for control environments
- Type hints encouraged

## Related Repositories

- Parent: [RKHS-KRONIC](https://github.com/Ed-Mehrez/RKHS-KRONIC)
- Sibling: [fsde-identifiability](https://github.com/Ed-Mehrez/fsde-identifiability)
- Sibling: [pomdp-koopman-control](https://github.com/Ed-Mehrez/pomdp-koopman-control)
