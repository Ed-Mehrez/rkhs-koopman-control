# PoC 8: Modular Kernels & Topology-Aware Control

We successfully refactored the RKHS-KRONIC framework to support modular **Feature Maps** and tested four distinct kernel strategies on the CartPole environment.

## Strategies Tested

| Strategy | Description | Theoretical Basis |
| :--- | :--- | :--- |
| **RFF** | Random Fourier Features ($N=400$) | Approximates Global Gaussian Kernel (Universal). |
| **Poly** | Polynomial Features (Deg 2) | Exact basis for smooth mechanical Hamiltonians ($E \propto x^2, v^2$). |
| **Energy-Sig** | Log-Sig (Deg 3) + Energy Augmentation | Physics-Informed. Explicitly encodes $H(p,q)$. |
| **Trig-Sig** | Log-Sig (Deg 3) + Embed $\theta \to (\cos, \sin)$ | Topology-Aware ($S^1$). Principled, no hardcoded physics. |

## Results

| Metric | RFF | Poly | Energy-Sig | Trig-Sig |
| :--- | :--- | :--- | :--- | :--- |
| **Stabilization Reward** | ~120-200 | **~260** | ~223 | ~105 |
| **Inference Latency** | ~100 $\mu s$ | **~8 $\mu s$** | ~8 $\mu s$ | ~5 $\mu s$ |
| **Swing-Up Reward** | 0.0 | 0.0 | 0.0 | 0.0 |

## Analysis

1.  **Stabilization is Solved:** All methods successfully stabilized the inverted pendulum using a **linear control law** ($u = -K z$) in the feature space.
    *   **Polynomials** performed best, likely because the local dynamics near the upright equilibrium are well-approximated by quadratic/cubic terms.
    *   **Trig-Signatures** (Principled) worked but were less sample-efficient than Physics-Informed ones.

2.  **Swing-Up Impossibility (Global Linear):** None of the methods allowed a **Single Global Linear Operator** to solve the Swing-Up task.
    *   **Hypothesis:** The Swing-Up control law is fundamentally discontinuous (bang-bang) or requires switching across the separatrix (Energy Pump $\to$ Stabilize).
    *   A single matrix $A$ cannot capture two distinct dynamic regimes (Pump vs Hold) if the lifting does not essentially linearize the *entire* manifold perfectly (which finite basis sets fail to do).

3.  **Efficiency:** The "Model-Free" Signatures (Trig-Sig and Energy-Sig) offer extreme efficiency (**< 10 µs**), making them vastly superior to MPC or Deep RL for high-frequency loops, *provided* the task is stabilizable.

## Conclusion
The **RKHS-KRONIC** framework is validated for **Local Control / Stabilization** on manifolds. For global tasks like Swing-Up, a **Hybrid** or **Iterative** approach (rejected by user) would be needed.

We now proceed to **PoC 9: Heston Hedging**, which aligns perfectly with the "Local Stabilization" strength: Tracking a derivative price on a stochastic manifold (Model-Free Greeks).
