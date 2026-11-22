#!/usr/bin/env python3
"""Audit of examples/experiment_local_koopman_sdre.py (README: "CartPole swing-up, 100 %").

The audited script is imported unchanged; its controller is re-instantiated with two
counters added (how often the random fallback fires, how far the cart travels).  The
environment, data collection, controller logic and success criterion (cos(theta) > 0.98
and |dtheta| < 0.5 for 100 consecutive steps) are the script's own.

Three questions:
  1. How often does the start-from-the-bottom run reach the criterion, and how long does
     it take, compared with the 2000-step loop in the script and with the pendulum's own
     time scale (small-oscillation period about the hanging position, 1.58 s = 79 steps)?
  2. How far does the cart travel?  The script's environment has no track limit.
  3. What happens with the standard +-2.4 m limit?

Run:  python examples/audit_cartpole_headline.py [n_seeds]
"""

import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.linalg import solve_continuous_are

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("headline", os.path.join(HERE, "experiment_local_koopman_sdre.py"))
headline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(headline)          # the script's __main__ block does not run on import


# small-oscillation period about the hanging position with the cart free, in steps of dt = 0.02:
# omega^2 = g / (l (4/3 - mp / (mc + mp))) with the script's g = 9.81, l = 0.5, mc = 1, mp = 0.1
PERIOD_STEPS = 2 * np.pi / np.sqrt(9.81 / (0.5 * (4 / 3 - 0.1 / 1.1))) / 0.02


class Instrumented(headline.LocalKoopmanSDRE):
    """Same control law as the audited class, with a counter on the random fallback."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n_fallback = 0
        self.n_calls = 0

    def get_control(self, state):
        self.n_calls += 1
        if len(self.X_data) < self.min_samples:
            return np.random.uniform(-30, 30)
        z = self.state_to_trig(state)
        X, U, Y = np.array(self.X_data), np.array(self.U_data), np.array(self.Y_data)
        dists = np.linalg.norm(X - z, axis=1)
        idx = np.argsort(dists)[: min(self.n_nearest, len(X))]
        X_s, U_s, Y_s = X[idx], U[idx], Y[idx]
        W = np.diag(np.exp(-dists[idx] ** 2 / (2 * self.kernel_width ** 2)))
        XU = np.hstack([X_s, U_s])
        try:
            AB = np.linalg.solve(XU.T @ W @ XU + 1e-4 * np.eye(6), XU.T @ W @ Y_s)
            A = (AB[:5].T - np.eye(5)) / self.dt
            B = AB[5:].T / self.dt
            Q = np.diag([0.0, 0.0, 50.0, 50.0, 0.0])
            R = np.eye(1) * 0.1
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P
            u = float((-K @ (z - np.array([0, 0, 1, 0, 0])))[0])
        except Exception:
            self.n_fallback += 1
            u = np.random.uniform(-15, 15)
        return np.clip(u, -30.0, 30.0)


def run(seed, max_steps, track=None):
    np.random.seed(seed)
    env = headline.SimpleCartPole(dt=0.02)
    ctrl = Instrumented(dt=0.02, kernel_width=0.3)
    for _ in range(40):                                   # the script's data-collection phase
        env.reset()
        env.state[2] = np.random.uniform(-np.pi, np.pi)
        obs = env.state.copy()
        for _ in range(50):
            u = np.random.uniform(-30, 30)
            nxt = env.step(u)
            ctrl.collect(obs, u, nxt)
            obs = nxt.copy()
    env.reset()
    env.state[2] = np.pi                                  # start from the bottom
    obs = env.state.copy()
    streak, xmax = 0, 0.0
    ctrl.n_fallback = ctrl.n_calls = 0
    for t in range(max_steps):
        u = ctrl.get_control(obs)
        nxt = env.step(u)
        ctrl.collect(obs, u, nxt)
        obs = nxt.copy()
        xmax = max(xmax, abs(obs[0]))
        if track is not None and abs(obs[0]) > track:
            return dict(ok=False, steps=t, why="left the track", xmax=xmax, fallback=ctrl.n_fallback / ctrl.n_calls)
        streak = streak + 1 if (np.cos(obs[2]) > 0.98 and abs(obs[3]) < 0.5) else 0
        if streak >= 100:
            return dict(ok=True, steps=t, why="", xmax=xmax, fallback=ctrl.n_fallback / ctrl.n_calls)
    return dict(ok=False, steps=max_steps, why="timeout", xmax=xmax, fallback=ctrl.n_fallback / ctrl.n_calls)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    budget = 8000
    print(f"No track limit (as in the audited script), budget {budget} steps = {budget * 0.02:.0f} s of simulated time")
    free = [run(s, budget) for s in range(n)]
    for s, r in enumerate(free):
        print(f"  seed {s}: success={r['ok']!s:5}  steps={r['steps']:5d} ({r['steps'] * 0.02:6.1f} s)  "
              f"max |cart position| = {r['xmax']:5.1f} m  random-fallback fraction {r['fallback']:.2f}  {r['why']}")
    ok = [r for r in free if r["ok"]]
    print(f"  -> {len(ok)}/{n} within {budget} steps; {sum(r['steps'] < 2000 for r in ok)}/{n} within the script's own 2000-step loop;")
    if ok:
        print(f"     median time to criterion {np.median([r['steps'] for r in ok]) * 0.02:.0f} s, "
              f"i.e. ~{np.median([r['steps'] for r in ok]) / PERIOD_STEPS:.0f} pendulum periods; "
              f"cart excursion {min(r['xmax'] for r in free):.1f} .. {max(r['xmax'] for r in free):.1f} m")
    print("\nStandard +-2.4 m track")
    lim = [run(s, budget, track=2.4) for s in range(n)]
    for s, r in enumerate(lim):
        print(f"  seed {s}: success={r['ok']!s:5}  steps={r['steps']:5d}  {r['why']}")
    print(f"  -> {sum(r['ok'] for r in lim)}/{n}")
