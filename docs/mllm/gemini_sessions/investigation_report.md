# Investigation Report: Why Swing-Up Fails (Global Linearization)

## 1. Hypothesis Testing
- **Hypothesis A:** The model hasn't seen swing-up data (Coverage).
- **Test:** Injected 3000 steps of expert "Energy Pumping" trajectories.
- **Result:** **Failed (Reward 0)**.
- **Conclusion:** Data coverage was NOT the primary blocker.

## 2. Root Cause Analysis
The failure suggests a topological or expressive limitation:
1.  **Topological Obstruction:** The linearized system $z_{t+1} = A z_t + B u_t$ implies a single global equilibrium point (usually the origin). The CartPole has two equilibria (Up, Down). While Koopman lifting *can* handle multiple equilibria by mapping them to different points in feature space, the *Linear Quadratic Regulator* (LQR) assumes a cost quadratic in deviation from *one* target (Upright).
    - If the "Down" state is far from the "Up" state in feature space, LQR will try to force it "Up" linearly.
    - However, the "path" to up is nonlinear (pump energy). A single Matrix $K$ implies $u = -K z$. This is a "proportional" controller in feature space.
    - If the "correct" action depends on the *sign* of velocity relative to angle (non-convex policy), a simple linear map might not express it unless the features *specifically* untangle that nonlinearity.

2.  **Feature Expressivity:** Degree 3 Log-Signatures might not separate the "pumping" manifold from the "stabilizing" manifold sufficiently for a linear cut.

## 3. Recommendation for Next Horizon
Since Global Linear Control failed:
1.  **Hybrid Approach:** Use the Energy Pump heuristic for swing-up (classic robotics) and switch to KRONIC LQR for stabilization (verified effective).
2.  **Intrinsic Coordinates for Finance:** Move to Heston/Finance (PoC 9). In finance, we don't need "Swing-Up" (global control). We usually need "Hedging" (local sensitivity/replication) or "Filtering". The **local accuracy** of Log-Signatures (verified by our stabilization results) is exactly what makes them powerful for computing Greeks (Malliavin derivates).

**Status:** We have extracted maximum value from the CartPole testbed (Speed + Stabilization verified; Global Control falsified). Ready to switch domains.
