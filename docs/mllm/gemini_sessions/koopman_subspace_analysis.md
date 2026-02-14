# Koopman Analysis: Subspace vs. Kernel

**Question:** Are we failing to find a good Koopman *subspace* (dictionary) or a good *kernel*?

**Answer:** We are failing to find a **Finite Invariant Subspace**.

## 1. The Subspace Problem (Finite Truncation)
The Koopman operator $\mathcal{K}$ is linear on the *infinite-dimensional* space of observables $\mathcal{F}$. To do control (LQR), we project this onto a *finite* basis $\psi(x) = [g_1(x), \dots, g_N(x)]^T$.
$$ \mathcal{K} \psi(x) \approx A \psi(x) $$
This approximation is exact **only if** $\text{span}(\psi)$ is an **Invariant Subspace** (i.e., $\mathcal{K} g_i \in \text{span}(\psi)$).

### Why Signatures Failed for Swing-Up
Signatures (and Log-Signatures) are fundamentally **Polynomials** on the path space.
- The CartPole Swing-Up involves global rotations: $x \in [0, 2\pi]$.
- The dynamics involve $f(x) \sim \sin(x), \cos(x)$ and energy pumping $E \sim \dot{\theta}^2$.
- **Key Algebra:** The derivative of a polynomial is a lower-order polynomial. BUT the derivative of a trigonometric function is another trigonometric function.
    - Taylor expanding $\sin(x)$ requires **infinite** polynomial terms to capture the periodicity $x \to x + 2\pi$.
    - A truncated degree-3 polynomial (Signature) cannot represent $\sin(x)$ globally. It diverges as $x \to \infty$ (or accumulates massive error wraps around $2\pi$).

**Conclusion:** A finite polynomial basis (Sig Degree 3) is **not an invariant subspace** for a globally periodic/trigonometric system. It is only good locally (Stabilization), which matches our results perfectly.

## 2. The Kernel Perspective
A "Kernel" $k(x, y) = \langle \psi(x), \psi(y) \rangle$ implicitly defines the feature space.
- We used a **Linear Kernel** on explicit Log-Sig features. This inherits the polynomial limits.
- A **Radial Basis Function (RBF) Kernel** ($e^{-\|x-y\|^2}$) corresponds to an *infinite* feature space (Fourier modes).
    - If we used Kernel-DMD (KDMD) with RBF, we *would* capture the periodicity.
    - **However**, KDMD does not give us explicit $A, B$ matrices for LQR easily (the "Pre-Image Problem").

## 3. Summary
We failed because **Degree 3 Log-Signatures do not span a Koopman Invariant Subspace for the global CartPole dynamics**.
- **Local Stabilization:** Valid (Polynomials approximate smooth functions locally).
- **Global Swing-Up:** Invalid (Polynomials fail to capture global topology/periodicity).

To fix this with Koopman, we would need:
1.  **Fourier-like features:** Features that encode periodicity ($\sin(x), \cos(x)$).
2.  **Higher Degree:** Much higher degree Signatures (expensive).
3.  **Kernel-based Control:** Switch to KRLS-Control (implicit $A, B$), but this loses the speed of the explicit LQR.

**Strategic Pivot:** This confirms that standard Signatures are best for **Local / Path-dependent** problems (Finance), not Global Topological problems (Robotics).
