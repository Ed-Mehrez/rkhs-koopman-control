# Malliavin Calculus: The Bridge to "Model-Free" Control

**User Question:** *"Were we talking about how the stochastic calc of variations might help us depend less on the model? Like akin to being more model-free?"*

**Answer:** **Yes.** This is the deepest theoretical advantage of the Signature/Log-Signature approach.

## 1. The "Model-Based" Trap
In standard Control (including our Koopman approach), we explicitly learn a "Transition Model":
$$ x_{t+1} \approx f(x_t, u_t) \quad \text{or} \quad z_{t+1} \approx A z_t + B u_t $$
We then differentiate this model ($\nabla_u f$) to find the optimal action. If the model is wrong (e.g., missed Swing-Up physics), the derivative is wrong, and control fails.

## 2. The "Model-Free" Alternative
In Model-Free RL (e.g., Policy Gradient), we don't learn $f$. We just sample paths and estimate:
$$ \nabla_\theta J \approx \mathbb{E} [ R(\tau) \nabla_\theta \log \pi(\tau) ] $$
This works but has **high variance** and requires millions of samples.

## 3. Malliavin Calculus: "Analytic Model-Free"
Stochastic Calculus of Variations (Malliavin Calculus) gives us a third way.
Instead of learning the *transition* $x \to x'$, we view the outcome $V(T)$ as a **functional of the driving noise** (the path).
$$ V(T) = F(W_t) $$
The **Malliavin Derivative** $D_t F$ tells us exactly how the outcome $V(T)$ changes if we "wiggle" the path at time $t$.
**Crucially:** For a system driven by Brownian motion, this derivative is related to the **Signature** of the path via the **Shuffle Product**.

### The "Greeks" Analogy (Finance)
In Finance, we don't predict the stock price $S_{t+1}$ (Model-Based). We compute the **Delta** ($\Delta = \partial C / \partial S$).
- Malliavin Calculus proves that $\Delta$ can be computed as an expectation of the Payoff weighted by a **Malliavin Weight** (Score Function).
- **We don't need the transition density $p(S'|S)$.** We just need the "Integration by Parts" formula on the path space.

### How this applies to CartPole/Heston
If we express our Value Function $V(x)$ in the **Log-Signature Basis**:
$$ V(x) \approx \langle \theta, \text{LogSig}(x) \rangle $$
Then the gradient $\nabla_u V$ (needed for control) can be computed **algebraically** using the Lie algebra properties of the Log-Sig.
- We rely on the **algebra of the basis** (Model-Free property of the features), not on the **parameters of the dynamical system** (Model-Based).
- This is "Model-Free" in the sense that we don't simulate "next states". We compute "sensitivities" directly from the path features.

**Conclusion:**
Moving to **Heston (Finance)** focuses purely on this "Hedging/Sensitivity" aspect. We stop trying to "Predict and Plan" (Robot Swing-Up) and start trying to "Replicate and Hedge" (Finance). This purely exploits the Malliavin nature of Log-Signatures.
