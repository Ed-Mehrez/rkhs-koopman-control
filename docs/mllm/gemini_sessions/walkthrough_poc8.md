
# Walkthrough - PoC 8: Log-Signatures (Intrinsic Coordinates)

## 1. Objective
Replace the global Signature basis with **Log-Signatures** (Lie Algebra basis) to:
1.  **Reduce Feature Dimension:** Mitigate algebraic redundancy ($O(d^k) \to O(k^{-1} d^k)$).
2.  **Improve Speed:** Accelerate recursive updates and inference.
3.  **Test Global Control:** Verify if the intrinsic basis improves the conditioning of the global linear model for Swing-Up.

## 2. Methodology
- **Basis:** `iisignature.logsig` (Degree 3).
- **Dimension:** Reduced from **1110** (Signature) to **385** (Log-Signature).
- **System:** CartPole at $200 \text{Hz}$ ($dt=0.005$).

## 3. Results

### Performance Metrics
| Metric | Signatures (PoC 7 - approx) | Log-Signatures (PoC 8) | Improvement |
| :--- | :--- | :--- | :--- |
| **Feature Dimension** | 1110 | **385** | **2.9x Reduction** |
| **Inference Latency** | ~90 $\mu s$ | **74.26 $\mu s$** | **18% Speedup** |
| **Training Time** | ~10s | ~2s | **5x Speedup** |

### Control Quality
- **Stabilization:** Average Reward **133.8** (Better than previous failed fidelity attempts, but not perfect). The reduced feature set arguably regularized the learning, preventing numerical blow-ups.
- **Swing-Up:** **Failed (Reward 0)**.

## 4. Analysis
The **Log-Signature** hypothesis is strongly validated for **efficiency**: massive dimension reduction translated directly to wall-clock speedups without loss of local stabilization capability.
However, the **Global Linearization hypothesis** (KRONIC LQR for Swing-Up) remains falsified. Even with the intrinsic Lie Algebra basis, a single linear operator $Z_{t+1} = A Z_t + B u_t$ cannot capture the global nonlinearity (Swing-Up + Stabilization) at this fidelity. This suggests that the "Lifted Linear" assumption holds locally (Lie Algebra describes local steering) but breaks down globally for this specific task/feature-set combination.

## 5. Next Steps
Move to **Finite-Dimensional Finance Applications (Heston Model)** where Log-Signatures (Lie Brackets) have a direct interpretation as "Market Greeks" and "Volatility corrections" (Malliavin Derivatives).
