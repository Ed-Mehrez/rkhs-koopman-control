# AGENTS.md

Context for AI coding agents working on this repository.

## Read first

This repository was audited in November 2026. The write-up is
`documentation/salvage_assessment_2026_11.md` in the parent repository
([RKHS-KRONIC](https://github.com/Ed-Mehrez/RKHS-KRONIC)); `README.md` here summarises what it
means for this split. The "Key Results" that used to be listed in this file (CartPole 100 %,
88x speedup, Kuramoto stable, cost reconstruction 0.12 %) are withdrawn. Do not restate them.

## Environment

Use the shared conda environment; do not create a new one:

```bash
conda activate rkhs-kronic-gpu
```

The `environment.yml` in this repo is for CI and collaborators. The audit script needs only
numpy, scipy and matplotlib.

## What is here

| Path | What it is |
|------|------------|
| `examples/experiment_local_koopman_sdre.py` | local-linear SDRE on a cart-pole, standalone |
| `examples/audit_cartpole_headline.py` | instrumented audit of the script above |
| `examples/experiment_kuramoto_scaling.py` | Kuramoto script, not re-run in the audit |
| `src/kgedmd_core.py`, `src/kronic_controller.py` | **legacy copies** with known defects (module docstrings) |

Current, tested estimator and controller code lives in the parent repository:
`src/kernel_generator.py` and `src/kernel_kronic.py`. The working cart-pole swing-up (bounded
track, 10 N) is there too: `experiments/salvage/cartpole_swingup.py`. Build on that, not on
the local-linear SDRE script in this repository.

## Rules

- A generator's largest-modulus eigenvalues are the least resolved. Never take
  `eigenvalues_[:k]` from the legacy estimator as "dominant modes".
- There are no universal kernel parameters. Bandwidth and rank are chosen by validation on
  held-out data, never by distance to an expected answer.
- No fallbacks that substitute made-up quantities (random actions, default gains). Count the
  failure and report it.
- A success rate is reported with its step budget, its state constraints (a cart-pole has a
  track) and a classical time scale next to it.
- A comparison between controllers gives both the same cost matrices and includes the trivial
  baselines.

## Conventions

- Python 3.10+, NumPy/SciPy
- Type hints encouraged

## Related repositories

- Parent: [RKHS-KRONIC](https://github.com/Ed-Mehrez/RKHS-KRONIC)
- Sibling: [fsde-identifiability](https://github.com/Ed-Mehrez/fsde-identifiability)
- Sibling: [pomdp-koopman-control](https://github.com/Ed-Mehrez/pomdp-koopman-control)
