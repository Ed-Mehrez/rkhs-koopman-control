# Theory: Explicit Features vs. Kernel Trick in Control

**Question:** Why not work directly in the Kernel Space (e.g., Signature Kernel) using Kernel Regression?

**Answer:** Because **Kernel Control** is computationally expensive ($O(N^3)$) and theoretically difficult compared to **Explicit Feature Control** ($O(d^3)$).

## 1. The "Explicit" Control Loop (What we are doing)
We map state $x \to z = \phi(x)$ (where $\phi$ is Log-Sig or RFF).
- **Model:** $z_{t+1} \approx A z_t + B u_t$. ($A, B$ are $d \times d$ matrices).
- **Control:** LQR gives $u_t = -K z_t$. ($K$ is $k \times d$).
- **Cost:** $O(d^3)$ to solve Riccati. $O(d^2)$ to infer.
- **Speed:** $d \approx 400$. Fast ($<100 \mu s$).

## 2. The "Implicit" Kernel Control Loop (What you are asking)
We use a Gram matrix $G_{ij} = k(x_i, x_j)$.
- **Model:** No explicit $A, B$. The operator is defined by weights $\alpha$ on the *dataset*: $\mathcal{K} \psi(x) = \sum_{i=1}^N \alpha_i k(x_i, x)$.
- **Control:** To do LQR, we must solve a "Kernel Riccati Equation".
    - This involves inverting the Gram matrix of the *entire training set* ($N \approx 13,000$).
    - **Cost:** $O(N^3)$. $13000^3 \approx \text{Trillions of ops}$. Intractable.
- **Inference:** $u_t = \sum_{i=1}^N \beta_i k(x_i, x_t)$. Requires evaluating kernel with *every* training point at every timestep. Slower.

## 3. Why RFF is the Solver
**Random Fourier Features (RFF)** are the bridge.
- They approximate the Kernel: $k(x, y) \approx \langle \text{RFF}(x), \text{RFF}(y) \rangle$.
- They give us **Explicit Features** $z = \text{RFF}(x)$.
- **Result:** We get the "Periodicity" and "Topology" of the Gaussian Kernel, but we keep the $O(d^3)$ speed of Explicit LQR.

## 4. The Signature Kernel?
The **Signature Kernel** $K_{sig}(X, Y)$ is just the kernel corresponding to the inner product of infinite signatures.
- **Pros:** Avoids truncation error.
- **Cons:** Still requires $O(N^3)$ for Kernel LQR.
- **Physics:** Still fundamentally based on iterated integrals (polynomial-like). It does not magically fix the periodicity issue of the CartPole unless we wrap the state space (e.g., using $\sin(\theta), \cos(\theta)$ as inputs before computing signatures).

**Conclusion:**
For real-time control with large datasets, **Explicit Features (RFF)** are superior to Implicit Kernel Methods. We will proceed with RFFs to capture the RBF geometry explicitly.
