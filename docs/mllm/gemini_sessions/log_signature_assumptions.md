# Log-Signatures in Control: Assumptions & Validity

## 1. What are Log-Signatures?
The Signature of a path $X$, denoted $S(X)$, lives in the tensor algebra $T((V))$. It is a sequence of all iterated integrals of the path.
The **Log-Signature**, $\log(S(X))$, lives in the **Lie Algebra** of the vector space, a much smaller subspace of the tensor algebra.

- **Signature:** "Global coordinates" (redundant). Encodes the path geometry linearly but inefficiently.
- **Log-Signature:** "Intrinsic coordinates" (compact). Encodes the same geometric information but only using the minimal number of generators (Lie brackets).

## 2. Key Assumptions for Validity

### A. The Shuffle Product Assumption (Algebraic)
*Assumption:* The path segments are processed as formal geometric objects where order matters, but "retracing" cancels out.
*Context:* The shuffle product identity states $\int dX^i \int dX^j + \int dX^j \int dX^i = \int dX^i X^j + \int dX^j X^i = X^i X^j$. The Signature contains all these redundant products. The Log-Signature effectively removes them, keeping only the anti-symmetric parts (Lie brackets like $[X^i, X^j]$) which correspond to "area" or "curvature" effects.
*Validity for CartPole:* **Valid.** The control affine system $\dot{x} = f(x) + g(x)u$ is driven by the geometric roughness of the path. The commutative parts (symmetric) correspond to pure time/parameterization changes which the physics might care about, but the *informational* content for control is largely in the non-commutative structure (Lie brackets).

### B. Smoothness / Rough Path Hypothesis
*Assumption:* The underlying continuous path is well-approximated by the discrete sequence of points at the given $dt$.
*Context:* Signature theory is built for Rough Paths (Brownian motion). For smooth control signals (500Hz), this is strictly stronger than needed, meaning it works *better*.
*Validity:* **Valid.** At 500Hz, the trajectories are very smooth. The Log-Signature approximation error is bounded by $O(\|X\|^{d+1})$, which is very small for short intervals.

### C. Tree-Like Equivalence (The "Blind Spot")
*Assumption:* The system does not distinguish between paths that are "tree-like" equivalent.
*Context:* Signatures (and Log-Signatures) cannot distinguish a path that goes $A \to B$ from one that goes $A \to B \to C \to B$. The excursion $B \to C \to B$ is "invisible" to the signature (it cancels out).
*Validity:* **Mostly Valid.** For a mechanical system with momentum (like CartPole), a rapid excursion $B \to C \to B$ *does* affect the state (energy dissipation, friction). However, if the time window is short and the path is monotonic in time (always moving forward in $t$), this "tree-like" cancellation doesn't happen for the time component. Since we include time (or monotonic step index) as a channel, the signature is **injective** (unique) and no information is lost.

## 3. Why it speeds things up
For a path in $\mathbb{R}^d$ truncated at degree $M$:
- **Signature Dimension:** $\approx d^M$ (Exponential).
    - Example ($d=5$, $M=3$): $\approx 5 + 25 + 125 = 155$.
- **Log-Signature Dimension:** Given by Witt's formula (much smaller).
    - Example ($d=5$, $M=3$): $\approx 5 + 10 + 30 = 45$.

**Reduction Factor:** Often 3x - 10x smaller feature vector.
**Compute Gain:** Recursive updates (RLS, Kalman) scale as $O(N_{feat}^2)$ or $O(N_{feat}^3)$. A 3x reduction in size means a **9x to 27x speedup** in compute.

## 4. Conclusion
For the CartPole task, **Log-Signatures are a mathematically valid and superior choice**. The inclusion of the "Time" channel ensures no tree-like ambiguity exists. The reduction in feature dimension directly addresses the $O(N^3)$ bottleneck of the RLS and LQR solvers without sacrificing geometric expressivity.
