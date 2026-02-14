
# Implementation Plan: MPPI Control for RKHS CartPole

## Problem
The CARE (LQR) controller leads to a "Runaway Cart" where the cart drives to infinity ($x > 15$) trying to pump energy into the angle. The linear approximation of the Bilinear term ($B_{eff}$) is insufficient for global swing-up.

## Solution: MPPI (Model Predictive Path Integral)
Instead of a static Gain Matrix $K$, we will use the learned model ($A, B, N$) to **simulate** the future in the feature space and optimize actions.

### Dynamic Model
$$ z_{t+1} = A z_t + B u_t + N (z_t \otimes u_t) $$
We can run this efficiently for $K=1000$ rollouts in parallel using GPU or vectorized Numpy.

### MPPI Algorithm
1.  Sample $K$ trajectories of control noise $\epsilon \sim \mathcal{N}(0, \Sigma)$.
2.  Propagate state $z$ forward $H$ steps using the learned Bilinear Model.
3.  Compute Cost $J(x_{traj})$ where $x = C_{dec} z$.
    *   Cost includes $Q_{angle}$, $Q_{x}$ (Soft Constraint), and $Q_{u}$.
4.  Compute weights $w \propto \exp(-J / \lambda)$.
5.  Update control sequence $U = \sum w \epsilon$.

## Changes to `poc_rkhs_cartpole.py`
1.  Define `rollout_kernel_model(z, U_seq)` function.
2.  Implement `mppi_control(z_curr, history)` function.
3.  Replace LQR loop.

## Expected Outcome
The controller will "see" that running away leads to high cost (via prediction) and will find the pumping strategy that keeps $x$ within bounds.
