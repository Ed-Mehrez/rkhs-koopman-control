# Implementation Plan: Principled Kernel Bandwidth Selection

## Problem Analysis

The previous session tried **Gibbs Kernel** (variable bandwidth) which failed because:
1. **Non-stationarity breaks linearity**: Koopman learning requires finding a *linear* operator `A` such that `z_next = A z`. If the metric changes spatially (Gibbs), the induced RKHS is warped, making linear prediction fundamentally harder.
2. **k-NN noise injection**: The k-NN bandwidth estimate is noisy, injecting variance into the feature map.
3. **Computational overhead**: k-NN queries at every step are slow.

The stationary **PeriodicKernel** with `l=4.0` was stable because it provides uniform metric structure.

## Proposed Solutions (Principled, Not Ad-Hoc)

### Approach 1: Cross-Validation on Dynamics Prediction Error (Recommended)

**Idea:** Instead of generic heuristics (median, k-NN), optimize bandwidth to minimize **one-step prediction error** on held-out data.

**Mathematical Justification:** We want `l` that minimizes:
$$\mathcal{L}(l) = \sum_{i \in \text{test}} \| z_{i+1} - A(l) z_i \|^2$$

where `A(l)` is the Koopman operator learned with bandwidth `l`.

**Implementation:**
```python
from sklearn.model_selection import KFold

def cv_bandwidth_selection(X, Y, kernel_class, l_candidates, n_folds=5):
    """Cross-validate bandwidth on dynamics prediction."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    
    for l in l_candidates:
        fold_errors = []
        for train_idx, val_idx in kf.split(X):
            # Learn A on train set with bandwidth l
            kernel = kernel_class(length_scale=l, ...)
            A = learn_koopman(X[train_idx], Y[train_idx], kernel)
            
            # Evaluate on validation
            Z_val = get_features(X[val_idx])
            Z_val_next = get_features(Y[val_idx])
            Z_pred = A @ Z_val.T
            error = np.mean((Z_val_next.T - Z_pred)**2)
            fold_errors.append(error)
        
        scores.append(np.mean(fold_errors))
    
    return l_candidates[np.argmin(scores)]
```

**Advantages:**
- Directly optimizes for the task (dynamics prediction)
- Principled (cross-validation is gold standard)
- Works with stationary kernels (no warping)

---

### Approach 2: Marginal Likelihood Optimization (Gaussian Process View)

**Idea:** Use the GP marginal likelihood (Rasmussen & Williams Ch 5.4) to select hyperparameters.

$$\log p(y|X,\theta) = -\frac{1}{2}y^T K_y^{-1} y - \frac{1}{2}\log|K_y| - \frac{n}{2}\log 2\pi$$

**Key Insight:** RKHS-EDMD is equivalent to GP regression. The marginal likelihood balances fit (first term) vs complexity (second term).

**Implementation:**
```python
from scipy.optimize import minimize

def negative_log_marginal_likelihood(log_l, X, Y, kernel_class, alpha=1e-6):
    l = np.exp(log_l)  # Ensure positivity
    kernel = kernel_class(length_scale=l, ...)
    K = kernel(X, X) + alpha * np.eye(len(X))
    
    try:
        L = np.linalg.cholesky(K)
        alpha_vec = scipy.linalg.cho_solve((L, True), Y)
        
        # Data fit term
        fit_term = 0.5 * np.sum(Y * alpha_vec)
        # Complexity term
        complexity = np.sum(np.log(np.diag(L)))
        # Constant
        const = 0.5 * len(X) * np.log(2*np.pi)
        
        return fit_term + complexity + const
    except np.linalg.LinAlgError:
        return 1e10  # Singular matrix

# Optimize
result = minimize(negative_log_marginal_likelihood, x0=np.log(4.0), 
                  args=(X, Y, PeriodicKernel), method='L-BFGS-B')
optimal_l = np.exp(result.x[0])
```

**Advantages:**
- Fully principled (Bayesian model selection)
- Automatic complexity regularization
- Standard in GP literature

---

### Approach 3: Anisotropic Bandwidth (Per-Dimension Scaling)

**Idea:** Instead of one global `l`, use different bandwidths for different state dimensions. Angular states (θ) may need different resolution than velocities.

**Mathematical Form:**
$$k(x, y) = \exp\left(-\sum_d \frac{(x_d - y_d)^2}{2 l_d^2}\right)$$

This is the **ARD kernel** (Automatic Relevance Determination) from Rasmussen & Williams.

**Implementation:**
```python
class AnisotropicRBF(BaseKernel):
    def __init__(self, length_scales):
        """length_scales: array of shape (n_dims,)"""
        self.length_scales = np.array(length_scales)
    
    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        # Scale each dimension
        X_scaled = X / self.length_scales
        Y_scaled = Y / self.length_scales
        
        from sklearn.metrics.pairwise import euclidean_distances
        dist_sq = euclidean_distances(X_scaled, Y_scaled, squared=True)
        return np.exp(-0.5 * dist_sq)
```

Then use CV or marginal likelihood to optimize `length_scales` (4 parameters for CartPole).

---

## Recommended Implementation Order

1. **Start with Cross-Validation** (Approach 1) - simplest and most robust
2. If more automation needed, add **Marginal Likelihood** (Approach 2)
3. If still struggling, try **Anisotropic ARD** (Approach 3)

## Verification Plan

1. Implement CV bandwidth selection
2. Compare selected `l` against manual `l=4.0`
3. Run 15s simulation and verify stability
4. If successful, document the principled methodology
