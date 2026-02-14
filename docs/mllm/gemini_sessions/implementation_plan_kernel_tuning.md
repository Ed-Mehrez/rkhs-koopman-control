# Implementation Plan - Kernel Parameter Tuning

## Goal
Address the control instability (drift at equilibrium) by tuning the Kernel parameters. The hypothesis is that the current `length_scale=1.0` is too broad, causing the model to smooth out the dynamics at the equilibrium point despite the anchor points.

## Proposed Changes

### `examples/proof_of_concept/poc_rkhs_cartpole.py`
1.  **Modify Kernel Parameters:**
    - Change `length_scale` from `1.0` to `0.5` (or `0.4`).
    - This sharpens the RBF/Periodic kernel, reducing the influence of far-away points on the equilibrium prediction.

2.  **Adjust Center Selection (Optional but Recommended):**
    - With a sharper kernel, we may need more centers to cover the space efficiently.
    - Increase `n_centers` from 300 to 500 (logic determines `n_components` in KernelPCA, here `dim_rkhs`).
    - Actually, `n_components` is set to 300. The Kernel Approximation uses `n_components`.
    - I will check where `n_components` is defined. It is `n_components=300`.
    - I will increase it to `400` if memory allows, or keep it 300. Let's start with just `length_scale`.

## Verification Plan

### Automated Verification
1.  **Drift Metric:** The script already prints `DRIFT CHECK: |A @ z_target|`. We expect this to decrease from 2.75 to < 1.0.
2.  **Simulation:** Run the simulation. If the pole balances (or drifts much slower), tuning was successful.

### Manual Verification
1.  Check the text output for "DRIFT CHECK" value.
2.  Visual check of `rkhs_cartpole_swingup.gif`.
