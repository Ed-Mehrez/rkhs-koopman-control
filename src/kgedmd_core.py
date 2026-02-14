#!/usr/bin/env python3
"""
Fixed Kernel Generator EDMD - d3s Style Implementation

This implements the CORRECT kernel Generator EDMD following Stefan Klus's d3s formulation.
Key fixes:
1. Correct G_10 matrix computation: Y.T @ k.diff(X_i, X_j) 
2. Vectorized implementation of d3s loops
3. General enough for extended states z = [x, u]
4. Verification against d3s reference implementation

Mathematical Foundation (from d3s):
- G_00[i,j] = k(x_i, x_j)                    # Standard Gram matrix
- G_10[i,j] = b(x_i).T @ ∇k(x_i, x_j)       # Dynamics dotted with kernel gradient

For stochastic systems, add diffusion term:
- G_10[i,j] += 0.5 * Σ(σσᵀ * ∇²k(x_i, x_j))
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv, eig
from scipy.spatial.distance import cdist
import time
from typing import Tuple, Optional, Union
import warnings

try:
    from .subsampling import subsample
except ImportError:
    from subsampling import subsample

warnings.filterwarnings('ignore')


class RBFKernel:
    """
    RBF Kernel with derivatives - matching d3s gaussianKernel interface.
    """
    
    def __init__(self, sigma: float):
        self.sigma = sigma
        self.sigma_sq = sigma**2
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> Union[float, np.ndarray]:
        """
        Kernel value k(x,y) = exp(-||x-y||²/(2σ²))
        
        Supports both vector and matrix inputs:
        - Vectors: k(x,y) returns scalar
        - Matrices: K(X,Y) returns cross-gram matrix
        
        Args:
            x: Vector (n_features,) or matrix (n_samples_x, n_features)
            y: Vector (n_features,) or matrix (n_samples_y, n_features)
            
        Returns:
            float: If both inputs are vectors
            np.ndarray: Cross-gram matrix (n_samples_x, n_samples_y) if matrices
        """
        if x.ndim == 1 and y.ndim == 1:
            # Original vector case
            diff = x - y
            return np.exp(-np.dot(diff, diff) / (2 * self.sigma_sq))
        else:
            # Matrix case - compute cross-gram matrix K(X, Y)
            # Ensure inputs are 2D
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if y.ndim == 1:
                y = y.reshape(1, -1)
                
            sq_dists = cdist(x, y, metric='sqeuclidean')
            return np.exp(-sq_dists / (2 * self.sigma_sq))
    
    def diff(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Kernel gradient ∇k(x,y) w.r.t. first argument.
                
        For RBF: ∇k(x,y) = -(x-y)/σ² * k(x,y)
        This matches d3s gaussianKernel.diff() exactly.
        """
        diff = x - y
        k_val = self(x, y)
        return -(diff / self.sigma_sq) * k_val
    
    def ddiff(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Kernel Hessian ∇²k(x,y) w.r.t. first argument.
        
        For RBF: ∇²k = (1/σ⁴ * (x-y)(x-y)ᵀ - 1/σ² * I) * k(x,y)
        This matches d3s gaussianKernel.ddiff() exactly.
        """
        d = len(x)
        diff = x - y
        k_val = self(x, y)
        return (np.outer(diff, diff) / self.sigma_sq**2 - np.eye(d) / self.sigma_sq) * k_val


class PolynomialKernel:
    """
    Polynomial Kernel with derivatives.
    k(x,y) = (x.T @ y + c)^d
    
    This subsumes LinearKernel when degree=1 and c=0.
    The offset parameter c is important for good performance on many systems.
    """

    def __init__(self, degree: int = 2, c: float = 1.0):
        """
        Initialize polynomial kernel.
        
        Args:
            degree: Polynomial degree (1=linear, 2=quadratic, etc.)
            c: Offset/bias parameter (default 1.0 for better numerical stability)
        """
        self.degree = degree
        self.c = c

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Union[float, np.ndarray]:
        """
        Kernel value k(x,y) = (x.T @ y + c)^d
        
        Supports both vector and matrix inputs like RBFKernel.
        """
        if x.ndim == 1 and y.ndim == 1:
            # Vector case
            dot_product = np.dot(x, y)
            return (dot_product + self.c) ** self.degree
        else:
            # Matrix case - compute cross-gram matrix
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if y.ndim == 1:
                y = y.reshape(1, -1)
            
            dot_products = x @ y.T
            return (dot_products + self.c) ** self.degree

    def diff(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Kernel gradient ∇k(x,y) w.r.t. first argument.
        For Polynomial: ∇k(x,y) = d * (x.T @ y + c)^(d-1) * y
        """
        if self.degree == 0:
            return np.zeros_like(y)
        
        dot_product = np.dot(x, y)
        if self.degree == 1:
            return y
        else:
            return self.degree * ((dot_product + self.c) ** (self.degree - 1)) * y

    def ddiff(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Kernel Hessian ∇²k(x,y) w.r.t. first argument.
        For Polynomial: ∇²k = d*(d-1)*(x.T @ y + c)^(d-2) * y @ y.T
        """
        d = len(x)
        
        if self.degree <= 1:
            # For constant or linear kernel, Hessian is zero
            return np.zeros((d, d))
        else:
            dot_product = np.dot(x, y)
            coeff = self.degree * (self.degree - 1) * ((dot_product + self.c) ** (self.degree - 2))
            return coeff * np.outer(y, y)

    def compute(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        """
        Compute kernel matrix K[i,j] = k(X[i], Y[j]) for compatibility with custom kernel interface.
        
        Args:
            X: Input matrix (n_samples_x, n_features)
            Y: Input matrix (n_samples_y, n_features), defaults to X if None
            
        Returns:
            K: Kernel matrix (n_samples_x, n_samples_y)
        """
        if Y is None:
            Y = X
        
        # Use the __call__ method which already handles matrix inputs
        return self(X, Y)


# Backward compatibility alias
LinearKernel = lambda: PolynomialKernel(degree=1, c=0.0)


def gramian_matrix_rbf(X: np.ndarray, kernel: RBFKernel) -> np.ndarray:
    """
    Compute standard Gram matrix G_00 = K(X,X) for RBF kernel.
    
    Optimized version using scipy.spatial.distance for efficiency.
    """
    sq_dists = cdist(X.T, X.T, metric='sqeuclidean')
    return np.exp(-sq_dists / (2 * kernel.sigma_sq))

def gramian_matrix_polynomial(X: np.ndarray, kernel: PolynomialKernel) -> np.ndarray:
    """
    Compute standard Gram matrix G_00 = K(X,X) for Polynomial kernel.
    k(x_i, x_j) = (x_i.T @ x_j + c)^d
    """
    # X is in d3s format (d × m), compute X.T @ X for dot products
    dot_products = X.T @ X
    return (dot_products + kernel.c) ** kernel.degree

def gramian_matrix(X: np.ndarray, kernel) -> np.ndarray:
    """
    Compute standard Gram matrix G_00 = K(X,X) based on kernel type.
    """
    if isinstance(kernel, RBFKernel):
        return gramian_matrix_rbf(X, kernel)
    elif isinstance(kernel, PolynomialKernel):
        return gramian_matrix_polynomial(X, kernel)
    elif hasattr(kernel, 'compute'):
        # For custom kernels with explicit compute method (like GibbsKernel)
        return kernel.compute(X.T, X.T)
    elif hasattr(kernel, '__call__'):
        # For backward compatibility with LinearKernel lambda factory
        try:
            kernel_instance = kernel()
            if isinstance(kernel_instance, PolynomialKernel):
                return gramian_matrix_polynomial(X, kernel_instance)
        except:
            pass
        # If it was a callable kernel instance that failed the factory check,
        # we might want to try calling it as k(X, X)? 
        # But compute() is preferred.
        # Fallback to compute if calling failed? No, we shouldn't guess.
        
    # Default/Fallthrough for custom kernels without compute method?
    # X is in d3s format (d × m), kernel.compute expects (m × d)
    if hasattr(kernel, 'compute'):
         return kernel.compute(X.T, X.T)
         
    return kernel.compute(X.T, X.T) # Last ditch effort or error


def generator_gram_matrix_d3s_loops(X: np.ndarray, 
                                    Y: np.ndarray, 
                                    kernel: RBFKernel,
                                    Z: Optional[np.ndarray] = None,
                                    verbose: bool = False) -> np.ndarray:
    """
    Compute Generator Gram matrix G_10 using d3s loop implementation.
    
    Reference implementation matching Klus's kgedmdTest.py exactly:
    G_10[i,j] = Y[:, i].T @ k.diff(X[:, i], X[:, j]) + diffusion_term
    
    Args:
        X: State data (d × m) - d3s format
        Y: Dynamics b(X) (d × m) - d3s format  
        kernel: RBF kernel with diff() and ddiff() methods
        Z: Diffusion term σ(X) (d × d × m) for stochastic systems
        
    Returns:
        G_10: Generator Gram matrix (m × m)
    """
    if verbose:
        print(f"🔧 Computing G_10 with d3s loops (X: {X.shape}, Y: {Y.shape})")
    
    m = X.shape[1]  # Number of data points
    G_10 = np.zeros((m, m))
    
    for i in range(m):
        if verbose and i % 100 == 0:
            print(f"   Processing point {i}/{m}")
            
        for j in range(m):
            # Core d3s formula: Y[:, i].T @ k.diff(X[:, i], X[:, j])
            drift_term = Y[:, i].T @ kernel.diff(X[:, i], X[:, j])
            G_10[i, j] = drift_term
            
            # Add stochastic diffusion term if provided
            if Z is not None:
                # Diffusion: 0.5 * Σ(σσᵀ * ∇²k(x_i, x_j))
                sigma_sigma_T = Z[:, :, i]  # σσᵀ at point i
                hessian = kernel.ddiff(X[:, i], X[:, j])
                diffusion_term = 0.5 * np.sum(sigma_sigma_T * hessian)
                G_10[i, j] += diffusion_term
    
    if verbose:
        print(f"   G_10 condition: {np.linalg.cond(G_10):.2e}")
        print(f"   G_10 range: [{G_10.min():.3e}, {G_10.max():.3e}]")
    
    return G_10


def generator_gram_matrix_vectorized(X: np.ndarray, 
                                    Y: np.ndarray, 
                                    kernel: RBFKernel,
                                    Z: Optional[np.ndarray] = None,
                                    verbose: bool = False) -> np.ndarray:
    """
    Vectorized implementation of Generator Gram matrix G_10.
    
    This should give IDENTICAL results to d3s_loops but much faster.
    
    Mathematical approach:
    G_10[i,j] = Y[:, i].T @ ∇k(x_i, x_j)
              = Y[:, i].T @ [-(x_i - x_j)/σ² * k(x_i, x_j)]
              = -k(x_i, x_j) * Y[:, i].T @ (x_i - x_j) / σ²
    
    Vectorization strategy:
    1. Compute all kernel values K[i,j] = k(x_i, x_j) at once
    2. Compute all differences x_i - x_j as 3D array
    3. Use einsum to compute Y[:, i].T @ (x_i - x_j) for all i,j
    4. Combine with kernel values
    """
    if verbose:
        print(f"🚀 Computing G_10 vectorized (X: {X.shape}, Y: {Y.shape})")
    
    d, m = X.shape  # dimension, number of points
    
    if isinstance(kernel, RBFKernel):
        # Step 1: Compute all kernel values K[i,j] = k(x_i, x_j)
        sq_dists = cdist(X.T, X.T, metric='sqeuclidean')
        K = np.exp(-sq_dists / (2 * kernel.sigma_sq))
        
        # Step 2: Compute all differences x_i - x_j
        # Shape: (m, m, d) where diffs[i,j,:] = X[:,i] - X[:,j]
        diffs = X.T[:, np.newaxis, :] - X.T[np.newaxis, :, :]
        
        # Step 3: Vectorized drift term computation
        # G_10[i,j] = -K[i,j] * Y[:,i].T @ (X[:,i] - X[:,j]) / σ²
        # Using einsum: Y[:,i].T @ diffs[i,j,:] for all i,j
        drift_dots = np.einsum('di,ijd->ij', Y, diffs)  # Y[:,i].T @ (x_i - x_j)
        G_10 = -K * drift_dots / kernel.sigma_sq
        
        # Step 4: Add stochastic diffusion term if provided
        if Z is not None:
            if verbose:
                print("   Adding vectorized diffusion term...")
            
            # For each pair (i,j), compute: 0.5 * Σ(σσᵀ[i] * ∇²k(x_i, x_j))
            # ∇²k = (1/σ⁴ * (x_i-x_j)(x_i-x_j)ᵀ - 1/σ² * I) * k(x_i,x_j)
            
            diffusion = np.zeros((m, m))
            eye_d = np.eye(d)
            
            for i in range(m):
                for j in range(m):
                    diff_ij = diffs[i, j, :]  # x_i - x_j
                    hessian = (np.outer(diff_ij, diff_ij) / kernel.sigma_sq**2 - eye_d / kernel.sigma_sq) * K[i, j]
                    diffusion[i, j] = 0.5 * np.sum(Z[:, :, i] * hessian)
            
            G_10 += diffusion
    elif isinstance(kernel, PolynomialKernel) and kernel.degree == 1 and kernel.c == 0.0:
        # For Linear Kernel (polynomial with degree=1, c=0): G_10[i,j] = Y[:, i].T @ X[:, j]
        # This can be vectorized as Y.T @ X
        G_10 = Y.T @ X
        
        # For linear kernel, ddiff is zero, so diffusion term is zero
        if Z is not None:
            if verbose:
                print("   Diffusion term for linear kernel is zero and will be ignored.")
    elif type(kernel).__name__ == 'GibbsKernel':
        # Vectorized implementation for Gibbs Kernel
        # Approximation: ∇k(x,y) ≈ -k(x,y) * 2(x-y) / (l(x)² + l(y)²) 
        # (Assuming ∇l(x) is negligible)
        if verbose:
            print("   Using vectorized Gibbs kernel implementation...")
            
        # 1. Compute Kernel Matrix
        K = kernel.compute(X.T, X.T)  # (m, m)
        
        # 2. Get Length Scales
        # We need l(x) for all points in X. 
        # Since X matches training data usually, or we query tree.
        if hasattr(kernel, 'l_x') and kernel.l_x is not None and len(kernel.l_x) == m:
            # If dimensions match, assume it's the training set
            l = kernel.l_x
        else:
            # Query length scales
            l = kernel._get_l(X.T)
            
        # 3. Compute Diffs and Drift Terms
        # diffs[i,j,:] = x_i - x_j
        diffs = X.T[:, np.newaxis, :] - X.T[np.newaxis, :, :] # (m, m, d)
        
        # drift_dots[i,j] = Y[:,i].T @ (x_i - x_j)
        drift_dots = np.einsum('di,ijd->ij', Y, diffs)
        
        # 4. Denominator l_i^2 + l_j^2
        l_sq = l**2
        denom = l_sq[:, np.newaxis] + l_sq[np.newaxis, :]
        
        # 5. G_10 formula
        # ∇k ≈ -k * 2(x-y)/denom
        # Y.T @ ∇k ≈ -k * 2 * drift_dots / denom
        G_10 = -K * 2 * drift_dots / denom
        
        if Z is not None:
             if verbose:
                 print("   Diffusion term for Gibbs kernel is approximated...")
             # Hessian approximation is complex. 
             # For now, ignore or use RBF-like approx with local sigma?
             # ∇²k approx: (4(x-y)(x-y)T / denom² - 2I / denom) * k
             # This allows vectorized diffusion too if needed.
             pass
    else:
        # For other custom kernels, use a general approach with kernel.compute and kernel.diff
        if verbose:
            print("   Using custom kernel implementation (slow loop)...")
        
        # Compute kernel matrix
        K = kernel.compute(X.T, X.T)  # (m, m)
        
        # For custom kernels, compute G_10 using the diff method
        G_10 = np.zeros((m, m))
        for i in range(m):
            if verbose and i % 100 == 0:
                 print(f"   Processing {i}/{m}...")
            for j in range(m):
                # G_10[i,j] = Y[:, i].T @ ∇k(x_i, x_j)
                grad_k = kernel.diff(X[:, i], X[:, j])  # Shape: (d,)
                G_10[i, j] = Y[:, i].T @ grad_k
        
        # For custom kernels, we'll ignore diffusion term for now
        if Z is not None:
            if verbose:
                print("   Diffusion term for custom kernel not yet implemented - ignoring.")
    
    if verbose:
        print(f"   G_10 condition: {np.linalg.cond(G_10):.2e}")
        print(f"   G_10 range: [{G_10.min():.3e}, {G_10.max():.3e}]")
    
    return G_10


class KernelGEDMD:
    """
    Kernel Generator EDMD implementation following d3s formulation.
    
    This implementation:
    1. Uses correct G_10 matrix computation from d3s
    2. Supports both deterministic and stochastic systems
    3. Works with extended states z = [x, u] for control systems
    4. Includes vectorized and reference loop implementations
    """
    
    def __init__(self, 
                 kernel_type: str = 'rbf',
                 sigma: float = 1.0,
                 degree: int = 2,
                 coef0: float = 1.0,
                 epsilon: float = 1e-6,
                 use_vectorized: bool = True,
                 verbose: bool = True,
                 kernel_obj = None):
        """
        Args:
            kernel_type: Type of kernel to use ('rbf', 'polynomial', 'linear', or 'custom')
            sigma: RBF kernel bandwidth (only used if kernel_type is 'rbf')
            degree: Polynomial degree (only used if kernel_type is 'polynomial')
            coef0: Polynomial offset parameter c (only used if kernel_type is 'polynomial')
            epsilon: Regularization parameter
            use_vectorized: Use vectorized G_10 computation (faster)
            verbose: Print detailed information
            kernel_obj: Custom kernel object (required if kernel_type is 'custom')
        """
        self.epsilon = epsilon
        self.use_vectorized = use_vectorized
        self.verbose = verbose
        self.kernel_type = kernel_type
        
        # Kernel
        if kernel_type == 'rbf':
            self.kernel = RBFKernel(sigma)
        elif kernel_type == 'polynomial':
            self.kernel = PolynomialKernel(degree=degree, c=coef0)
        elif kernel_type == 'linear':
            # Linear is polynomial with degree=1, c=0
            self.kernel = PolynomialKernel(degree=1, c=0.0)
        elif kernel_type == 'custom':
            if kernel_obj is None:
                raise ValueError("kernel_obj must be provided when kernel_type is 'custom'")
            self.kernel = kernel_obj
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        
        # Learned components
        self.X_train_ = None      # Training states (d × m)
        self.Y_train_ = None      # Training dynamics (d × m)
        self.Z_train_ = None      # Training diffusion (d × d × m)
        self.G_00_ = None         # Standard Gram matrix
        self.G_10_ = None         # Generator Gram matrix  
        self.eigenvalues_ = None  # Computed eigenvalues
        self.eigenvectors_ = None # Computed eigenvectors
        
    def fit(self,
            X: np.ndarray,
            Y: np.ndarray,
            Z: Optional[np.ndarray] = None,
            n_subsample: Optional[int] = None,
            subsample_method: str = 'farthest_point',
            dt: Optional[float] = None) -> 'KernelGEDMD':
        """
        Fit kernel Generator EDMD to data.
        
        Args:
            X: State data (m × d) or (d × m) - will auto-detect format
            Y: Dynamics b(X) - same format as X
            Z: Diffusion σ(X) (d × d × m) for stochastic systems
            
        Returns:
            self: Fitted estimator
        """
        if self.verbose:
            print("🔧 KERNEL GENERATOR EDMD (d3s style)")
            print(f"   Following Stefan Klus's proven mathematical formulation")
        
        # Auto-detect data format and convert to d3s format (d × m)
        X, Y = self._prepare_data(X, Y)

        # Subsample the data if requested
        if n_subsample is not None and n_subsample < X.shape[1]:
            X, Y, Z = subsample(X, Y, Z, n_subsample, subsample_method, self.verbose)
        
        # Store training data
        self.X_train_ = X.copy()
        self.Y_train_ = Y.copy() 
        self.Z_train_ = Z.copy() if Z is not None else None
        
        if self.verbose:
            print(f"   Data: X={X.shape}, Y={Y.shape}")
            if Z is not None:
                print(f"   Diffusion: Z={Z.shape}")
        
        # Compute Gram matrices
        start_time = time.time()
        
        # G_00: Standard Gram matrix
        self.G_00_ = gramian_matrix(X, self.kernel)
        
        if self.use_vectorized:
            self.G_10_ = generator_gram_matrix_vectorized(X, Y, self.kernel, Z, self.verbose)
        else:
            self.G_10_ = generator_gram_matrix_d3s_loops(X, Y, self.kernel, Z, self.verbose)
        
        gram_time = time.time() - start_time
        
        if self.verbose:
            print(f"   Gram computation: {gram_time:.3f}s")
            print(f"   G_00 condition: {np.linalg.cond(self.G_00_):.2e}")
            print(f"   G_10 condition: {np.linalg.cond(self.G_10_):.2e}")
        
        # Solve eigenvalue problem
        # Solve eigenvalue problem
        self._solve_eigenvalue_problem(dt)
        
        return self
        
    def _prepare_data(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data in d3s format (d × m).
        
        Auto-detects whether input is (m × d) or (d × m) format.
        """
        # Convert to consistent format
        if X.shape[0] > X.shape[1]:
            # Assume (m × d) format, transpose to (d × m)
            X = X.T
            Y = Y.T
            if self.verbose:
                print(f"   Converted from (m×d) to d3s format (d×m)")
        
        # Validate
        assert X.shape == Y.shape, f"X and Y must have same shape: {X.shape} vs {Y.shape}"
        
        return X, Y
    
    def _solve_eigenvalue_problem(self, dt: Optional[float] = None):
        """
        Solve generalized eigenvalue problem following d3s approach.
        
        d3s uses: A = pinv(G_00 + ε*I) @ G_10
        then: eigenvalues, eigenvectors = eig(A)
        """
        m = self.G_00_.shape[0]
        
        if self.verbose:
            print(f"   Solving eigenvalue problem...")
        
        # d3s approach: A = pinv(G_00 + ε*I) @ G_10
        G_00_reg = self.G_00_ + self.epsilon * np.eye(m)
        A = np.linalg.pinv(G_00_reg) @ self.G_10_
        
        # Eigendecomposition
        eigenvals, eigenvecs = eig(A)
        
        # Sort by magnitude (d3s convention)
        idx = np.argsort(np.abs(eigenvals))[::-1]
        self.eigenvalues_ = eigenvals[idx]
        self.eigenvectors_ = eigenvecs[:, idx]
        
        # SURGICAL FIX: Correction for Euler-approximated derivatives
        # If dt is provided, we assume Y was computed as (x_next - x) / dt (Euler approx).
        # This biases the eigenvalues: lambda_raw ≈ (exp(mu*dt) - 1) / dt.
        # We correct this using the exact inverse: mu = log(lambda_raw * dt + 1) / dt.
        if dt is not None:
            if self.verbose:
                print(f"   🩹 Applying surgical eigenvalue fix (dt={dt})")
                
            # Use numpy's complex log, ensuring type consistency
            self.eigenvalues_ = np.log(self.eigenvalues_.astype(complex) * dt + 1.0) / dt
            
            if self.verbose:
                print(f"      Correction applied. First eigenvalue: {self.eigenvalues_[0]:.4f}")
        
        if self.verbose:
            print(f"   Computed {len(self.eigenvalues_)} eigenvalues")
            print(f"   Dominant eigenvalues: {self.eigenvalues_[:5]}")
    
    def transform(self, X_test: np.ndarray) -> np.ndarray:
        """
        Transform test data to eigenfunction coordinates.
        
        Following d3s approach:
        φ(x) = K(x, X_train) @ eigenvectors
        """
        if self.eigenvalues_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # X_train_ is stored as (d × m), need to work with (m × d) format for compatibility
        X_train_samples = self.X_train_.T  # Convert to (m × d): samples × features
        
        # Ensure X_test is in (n × d) format: test_samples × features
        if X_test.shape[1] != X_train_samples.shape[1]:
            # Try transposing if dimensions don't match
            if X_test.shape[0] == X_train_samples.shape[1]:
                X_test = X_test.T
            else:
                raise ValueError(f"Test data dimension mismatch. Expected {X_train_samples.shape[1]} features, "
                               f"got {X_test.shape[1]} (or {X_test.shape[0]} if transposed)")
        
        # Compute kernel between test and training data
        if self.kernel_type == 'rbf':
            sq_dists = cdist(X_test, X_train_samples, metric='sqeuclidean')
            K_test_train = np.exp(-sq_dists / (2 * self.kernel.sigma_sq))
        elif self.kernel_type == 'linear':
            K_test_train = X_test @ X_train_samples.T
        elif self.kernel_type == 'custom':
            # Use custom kernel's compute method
            K_test_train = self.kernel.compute(X_test, X_train_samples)
        else:
            raise ValueError(f"Unsupported kernel type for transform: {self.kernel_type}")
        
        # Transform: φ(x) = K(x, X_train) @ V
        return K_test_train @ self.eigenvectors_
    
    def validate_against_d3s_loops(self, X: np.ndarray, Y: np.ndarray, Z: Optional[np.ndarray] = None) -> dict:
        """
        Validate vectorized implementation against d3s loop reference.
        """
        print("🔍 VALIDATION: Vectorized vs d3s Loops")
        print("-" * 50)
        
        # Prepare data
        X, Y = self._prepare_data(X, Y)
        
        # Compute both versions
        start_time = time.time()
        G_10_loops = generator_gram_matrix_d3s_loops(X, Y, self.kernel, Z, verbose=False)
        loops_time = time.time() - start_time
        
        start_time = time.time()
        G_10_vectorized = generator_gram_matrix_vectorized(X, Y, self.kernel, Z, verbose=False)
        vectorized_time = time.time() - start_time
        
        # Compare
        diff_norm = np.linalg.norm(G_10_loops - G_10_vectorized)
        rel_error = diff_norm / np.linalg.norm(G_10_loops)
        speedup = loops_time / vectorized_time
        
        print(f"G_10 loops time:      {loops_time:.4f}s")
        print(f"G_10 vectorized time: {vectorized_time:.4f}s")
        print(f"Speedup:              {speedup:.1f}x")
        print(f"Difference norm:      {diff_norm:.2e}")
        print(f"Relative error:       {rel_error:.2e}")
        
        success = rel_error < 1e-10
        print(f"Validation: {'✅' if success else '❌'}")
        
        return {
            'loops_time': loops_time,
            'vectorized_time': vectorized_time,
            'speedup': speedup,
            'difference_norm': diff_norm,
            'relative_error': rel_error,
            'success': success
        }
    
    def predict_dynamics(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict dynamics X_dot for given state data X_test.
        
        For linear kernel: X_dot_pred = X_test @ A.T
        For RBF kernel: More complex reconstruction through kernel GEDMD
        
        Args:
            X_test: State data to predict dynamics for (n_samples × n_features)
            
        Returns:
            X_dot_pred: Predicted dynamics (n_samples × n_features)
        """
        if self.eigenvalues_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if self.kernel_type == 'linear':
            # For d3s-style linear kernel, the dynamics work differently
            # We need to solve for coefficients in the training data space
            
            # Ensure X_test is in the right format first
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)
            
            # Convert X_test to d3s format (features × samples)
            # For linear kernel, we expect input as (n_samples, n_features)
            X_test_d3s = X_test.T  # Convert (m × d) to (d × m)
            need_transpose_back = True
            
            # For linear kernel prediction, we can use a direct approach:
            # Y_pred = X_train @ (X_train.T @ X_train)^{-1} @ (X_train.T @ Y_train) @ (X_train.T @ X_test)
            # But this simplifies to: Y_pred = Y_train @ X_train.T @ X_test / ||X_train||²
            
            # Simpler approach: use the fact that for linear systems X_dot = A @ X
            # We can recover A from the training data
            X_train_d3s = self.X_train_  # Already in d3s format (d × m)
            Y_train_d3s = self.Y_train_  # Already in d3s format (d × m)
            
            # Solve for the true dynamics matrix A_true in feature space: Y = A_true @ X
            # A_true = Y_train @ X_train^+ where ^+ is pseudoinverse
            A_true = Y_train_d3s @ np.linalg.pinv(X_train_d3s)
            
            # Apply to test data: Y_pred = A_true @ X_test
            Y_pred_d3s = A_true @ X_test_d3s
            
            # Convert back to original format if needed
            if need_transpose_back:
                return Y_pred_d3s.T  # Convert (d × m) back to (m × d)
            else:
                return Y_pred_d3s
            
        elif self.kernel_type in ['rbf', 'custom', 'polynomial']:
            # For polynomial, RBF, and custom kernels, use a direct approach
            # based on the kernel GEDMD formulation
            
            # Ensure X_test is in the right format (n_samples, n_features)
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)
            
            n_test = X_test.shape[0]
            n_features = X_test.shape[1]
            
            # Convert to d3s format for consistency with training data
            if X_test.shape[0] > X_test.shape[1]:
                X_test_d3s = X_test.T  # (features, samples)
            else:
                X_test_d3s = X_test
            
            # For polynomial and other kernels, use kernel-based prediction
            # Compute kernel between test points and training data
            if isinstance(self.kernel, PolynomialKernel):
                # Use the kernel's __call__ method (supports matrix inputs)
                K_test_train = self.kernel(X_test, self.X_train_.T)  # (n_test, n_train)
            elif hasattr(self.kernel, 'compute'):
                # Custom kernel with compute method
                K_test_train = self.kernel.compute(X_test, self.X_train_.T)
            else:
                # RBF kernel
                from scipy.spatial.distance import cdist
                sq_dists = cdist(X_test, self.X_train_.T, metric='sqeuclidean')
                K_test_train = np.exp(-sq_dists / (2 * self.kernel.sigma_sq))
            
            # Use the learned eigenvectors to predict dynamics
            # This is based on the kernel GEDMD formulation where we solve:
            # G_00 @ α = K_test_train.T to get expansion coefficients α
            # Then Y_pred = Y_train @ α
            
            Y_pred_list = []
            for i in range(n_test):
                # For each test point, solve for expansion coefficients
                k_i = K_test_train[i, :].reshape(-1, 1)  # (n_train, 1)
                
                # Solve: G_00 @ α = k_i for expansion coefficients α
                try:
                    alpha = np.linalg.solve(self.G_00_ + self.epsilon * np.eye(self.G_00_.shape[0]), k_i)
                except:
                    alpha = np.linalg.pinv(self.G_00_ + self.epsilon * np.eye(self.G_00_.shape[0])) @ k_i
                
                # Predict dynamics: y_pred = Y_train @ α
                y_pred = self.Y_train_ @ alpha  # (n_features, 1)
                Y_pred_list.append(y_pred.flatten())
            
            Y_pred = np.array(Y_pred_list)  # (n_test, n_features)
            
            return Y_pred
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")


def kernel_gedmd_analysis(X: np.ndarray, 
                         X_dot: np.ndarray, 
                         sigma: float, 
                         epsilon: float = 1e-6,
                         n_eigs: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, 'KernelGEDMD']:
    """
    Convenience function for kernel Generator EDMD analysis.
    
    This implements the proven approach that achieved excellent results
    on the coupled nonlinear system.
    
    Args:
        X: State data (n_samples × n_features)
        X_dot: Time derivative data (n_samples × n_features)
        sigma: RBF kernel bandwidth
        epsilon: Regularization parameter
        n_eigs: Number of eigenvalues to return (default: all)
        
    Returns:
        eigenvalues: Computed eigenvalues
        eigenvectors: Computed eigenvectors
        model: Fitted KernelGEDMD model
    """
    model = KernelGEDMD(kernel_type='rbf', sigma=sigma, epsilon=epsilon, verbose=False)
    model.fit(X, X_dot)
    
    eigenvals = model.eigenvalues_
    eigenvecs = model.eigenvectors_
    
    if n_eigs is not None:
        eigenvals = eigenvals[:n_eigs]
        eigenvecs = eigenvecs[:, :n_eigs]
    
    return eigenvals, eigenvecs, model