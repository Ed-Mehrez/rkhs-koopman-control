# RKHS Koopman Control

Example scripts for local-linear SDRE control (cart-pole, Kuramoto, double well), split from
[RKHS-KRONIC](https://github.com/Ed-Mehrez/RKHS-KRONIC) in February 2026.

## Status, November 2026

This repository was audited together with its parent. The full write-up is
`documentation/salvage_assessment_2026_11.md` in RKHS-KRONIC. What it means here:

**The earlier headline numbers in this README are withdrawn.**

| Earlier claim | Audit result | How to reproduce |
|---|---|---|
| CartPole swing-up, 100 % (50/50) | 1/6 within the script's own 2000-step loop; 5/6 within 8000 steps (160 s of simulated time), median 78 s, about 49 pendulum periods; the cart travels 3.9 to 6.1 m on a track with no limit. With a standard ±2.4 m track: 0/6. | `python examples/audit_cartpole_headline.py 6` |
| Kuramoto 20-D, stable for 10k steps | Not re-run. The parent repository's own May 2026 note records it as unseeded and without baselines; its seeded rerun gave 0/3 against 3/3 for frequency cancellation. | |
| Cost reconstruction, 0.12 % mean error | The quantity is kernel ridge regression of the state evaluated back on the state; it is not a cost in eigenfunction coordinates. | |
| "88x faster" | A k-nearest-neighbour restriction of a local regression. Not re-measured. | |

What the cart-pole script actually is: a local-linear model fitted to the 100 nearest
transitions, an LQR on it with a cost on the angle only, applied from the hanging position.
Its random fallback never fires. It reaches the upright criterion by eventually passing close
enough for the local LQR to hold, not by a swing-up strategy.

**The `src/` package here is a legacy copy.** `src/kgedmd_core.py` and
`src/kronic_controller.py` carry the defects described in the assessment (see their module
docstrings). Until this audit they could not be imported at all, because `subsampling.py` had
not been copied in the split; it has been added so they import. The usage snippet that used to
be in this README (`KernelGEDMD(kernel, regularization=...)`, `KRONICController(kgedmd, Q=...,
R=...)`) never matched their signatures.

**For working code use the parent repository:** `src/kernel_generator.py` (reduced-rank kernel
gEDMD with residual-verified eigenpairs) and `src/kernel_kronic.py` (KRONIC control with
jointly learned drift and input generators). There, a pendulum is swung up from data in a few
swings with torque limited to 10 % of gravity, using a kernel-discovered conserved quantity.

**The cart-pole swing-up now works there, on a bounded track.** The same plant as the script
audited above, but with the force limited to 10 N (the script allowed 30 N) and a ±2.4 m track
(the script had none): the controller swings up and holds from 101/101 starts it was not
designed on, upright and at rest within 5.0 s (about three pendulum periods), with every
quantity it uses learned from 2,500 random samples by one kernel generator. The saturated
oracle LQR alone manages 25/101. See `experiments/salvage/cartpole_swingup.py`,
`tests/test_cartpole_swingup.py` and section 3.5 of the assessment in the parent repository.
It is a different method from the script here (a conserved quantity, a momentum chain and an
LQR catch, not a local-linear SDRE), so it replaces the withdrawn claim rather than restoring
it.

## What runs

| Script | State |
|---|---|
| `examples/experiment_local_koopman_sdre.py` | runs standalone (writes a GIF) |
| `examples/audit_cartpole_headline.py` | runs standalone; the audit of the script above |
| `examples/experiment_kuramoto_scaling.py` | imports resolve; not re-run |
| `examples/proof_of_concept/poc_rkhs_cartpole_kgedmd.py`, `poc_rkhs_cartpole_principled.py` | imports resolve; not re-run; built on the legacy estimator |
| `examples/experiment_rkhs_kronic_stabilization.py`, `examples/proof_of_concept/poc_double_well_control.py` | need the `environments` package from the parent repository |
| `examples/proof_of_concept/poc_rkhs_cartpole.py` | needs `src/models` from the parent repository |

```bash
pip install numpy scipy matplotlib
python examples/audit_cartpole_headline.py 6      # about 12 minutes
```

## Layout

```
src/        legacy copies of the estimator and controller (see module docstrings)
examples/   scripts listed above
docs/       theory notes and exported assistant sessions from early 2026, not reviewed
```

## References

- Klus, Nüske, Hamzi. Kernel-based approximation of the Koopman generator and Schrödinger
  operator. *Entropy* 22 (2020).
- Kaiser, Kutz, Brunton. Data-driven discovery of Koopman eigenfunctions for control.
  *Mach. Learn.: Sci. Technol.* 2 (2021).
- Williams, Kevrekidis, Rowley. A data-driven approximation of the Koopman operator:
  extending dynamic mode decomposition. *J. Nonlinear Sci.* 25 (2015).

## License

MIT License
