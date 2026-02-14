# Staged Koopman Control & Tensor Decomposition Walkthrough

## 1. Overview
This walkthrough details the implementation of a **Staged Learning** approach combined with **Tensor-Based Model Reduction** for the RKHS-KRONIC control framework.

**Goal:** Identify Bilinear Koopman dynamics $\dot{z} = Az + Bu + N(z \otimes u)$ that are suitable for control, avoiding the "loss of control authority" common in naive SVD approaches.

---

## 2. Key Implementations

### A. Staged Learning
We replaced the monolithic regression with a 3-stage process to isolate dynamics:

| Stage | Data Type | Goal | Resulting Metric |
|-------|-----------|------|------------------|
| **1** | Autonomous ($u=0$) | Learn $A$ (Drift) | **$R^2 \approx 0.9999$** |
| **2** | Equilibrium ($x \approx 0$) | Learn $B$ (Control) | **$R^2 \approx 0.948$** |
| **3** | Full Controlled | Learn $N$ (Bilinear) | **$R^2 \approx 0.980$** |

**Equilibrium Anchoring:** We added 200 copies of the exact fixed point $(x=0, u=0) \to x=0$ to the Stage 1 dataset. This reduced the model drift at equilibrium from $|A z^*| \approx 3.06$ to $2.75$.

### B. Tensor-Based Model Reduction
We implemented a **Control-Aware HOSVD** approach effectively.

**The Problem:** Naive SVD on the autonomous dynamics $A$ often discards the subspace where control $B$ acts (if it's orthogonal to natural dynamics).
**The Solution:** Perform SVD on the augmented tensor unfolding $[A \;|\; N \;|\; B]$.

**Results:**
- **Full Model $|B|$ norm:** 2.20
- **Naive SVD $|B_{red}|$ norm:** 0.26 (88% Loss of Authority)
- **Tensor SVD $|B_{red}|$ norm:** **2.20 (100% Preservation)**

This confirms the tensor approach successfully preserves control authority in the reduced latent space.

---

## 3. Validation Results

### Simulation
- **Control Gain:** The Tensor-reduced model yields much stronger feedback gains ($|K| \approx 1.3$) compared to the previous weak gains ($|K| \approx 0.1$).
- **Stability:** The Pole still drifts (slowly or quickly depending on tuning).
- **Diagnosis:** The residual drift at the target state ($|A z^*| \approx 2.75$) implies the learned model "thinks" the upright state is moving. The controller fights this phantom drift, destabilizing the real system.

### Comparison Plot
![Frequency Comparison](/home/ed/SynologyDrive/Documents/Research/P&E_Research/rkhs_kronic/rkhs_cartpole_freq_compare.png)

### Animation
High-frequency control attempt:
![Swing Up](/home/ed/SynologyDrive/Documents/Research/P&E_Research/rkhs_kronic/rkhs_cartpole_swingup.gif)

---

## 4. Next Steps

1.  **Drift Cancellation:** Implement explicit drift subtraction in the control law or enforce hard constraints on $A$ during learning (Constrained Least Squares).
2.  **Kernel Tuning:** The breakdown of the fixed point might be due to the Periodic Kernel's length scale.
3.  **Hybrid Approach:** Use the learned $B$ and $N$, but replace learned $A$ with linearized physics for the local region to guarantee stability.

## 5. Files
- [poc_rkhs_cartpole.py](file:///home/ed/SynologyDrive/Documents/Research/P&E_Research/rkhs_kronic/examples/proof_of_concept/poc_rkhs_cartpole.py) - Main detailed implementation
- [koopman_tensor_decomposition.md](file:///home/ed/SynologyDrive/Documents/Research/P&E_Research/rkhs_kronic/documentation/koopman_tensor_decomposition.md) - Theoretical basis
