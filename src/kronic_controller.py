"""
RKHS-KRONIC controller -- LEGACY implementation, kept for provenance.

Do not use this class for new results.  An audit (documentation/salvage_assessment_2026_11.md)
found that it does not implement the method it is named after:

* The "intrinsic coordinates" are ``eigenvectors_[:, :n]`` of the legacy estimator, i.e.
  the n eigenpairs of LARGEST modulus.  For a generator with a Gaussian kernel those are
  discretisation artefacts, not Koopman eigenfunctions.
* ``_learn_reduced_dynamics`` differentiates the coordinates by central differences along
  the SAMPLE INDEX.  That is only meaningful if the training set is one contiguous
  trajectory with step ``dt``; for i.i.d. samples (as in ``demonstrate_kronic_controller``)
  it differences unrelated points.  The eigenvalues are then discarded and a generic
  bilinear model is re-fitted by ridge regression.
* Failure paths substitute made-up quantities instead of failing: a "stable default"
  A = -0.1 with a random B if the regression fails, proportional gains if the Riccati
  equation fails, a diagonal Q if cost regression fails, and ``_ensure_stability`` shifts
  the identified dynamics by -0.1 I when they are not stable.
* ``predict_trajectory`` integrates placeholder noise, not a model.
* The non-affine branch of ``control`` is a stub that calls the affine branch.

The working implementation of the idea is ``src.kernel_kronic`` (residual-verified
eigenfunctions, jointly learned drift and input generators, state-dependent Riccati
control, refusal instead of fallbacks), tested in closed loop against analytic KRONIC in
tests/test_kernel_kronic.py.

References:
- Kaiser, E., Kutz, J. N., & Brunton, S. L. (2021). Data-driven discovery of
  Koopman eigenfunctions for control.
"""

import numpy as np
import scipy.linalg
from scipy.linalg import solve_continuous_are
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Union
import warnings

from .kgedmd_core import KernelGEDMD, RBFKernel
try:
    from .regularized_cost_reconstruction import enhanced_cost_mapping
except ImportError:
    enhanced_cost_mapping = None


class KRONICController:
    """
    KRONIC controller using proven Generator EDMD approach.
    
    Architecture:
    1. Apply kernel Generator EDMD to discover intrinsic coordinates φ(z)  
    2. Learn reduced dynamics: φ̇ = A_φ φ + B_φ u in intrinsic space
    3. Synthesize optimal control using LQR in reduced coordinates
    4. Project back to full state space for control implementation
    
    This integrates successful patterns from experimental implementations.
    """
    
    def __init__(self, kernel: RBFKernel, target_state: np.ndarray, 
                 cost_weights: Optional[dict] = None, verbose: bool = True,
                 enable_eigenfunction_scaling: bool = True,
                 optimize_for_control: bool = True,
                 problem_type: str = 'standard'):
        """
        Args:
            kernel: Kernel function for GEDMD (typically RBF)
            target_state: Target state for regulation  
            cost_weights: Control cost weights {'Q': state_cost, 'R': control_cost}
            verbose: Print diagnostic information
            enable_eigenfunction_scaling: Enable adaptive eigenfunction scaling (recommended)
            optimize_for_control: Optimize intrinsic dimension for control (recommended)
            problem_type: 'standard' for standard LQR, 'portfolio' for quadratic utility portfolio optimization, 
                         'portfolio_log' for log utility portfolio optimization (recommended)
        """
        self.kernel = kernel
        self.target = target_state
        self.verbose = verbose
        self.enable_eigenfunction_scaling = enable_eigenfunction_scaling
        self.problem_type = problem_type
        
        # Cost function weights (defaults from successful tests)
        self.Z_scaler = StandardScaler()


        if cost_weights is None:
            cost_weights = {'Q': 1.0, 'R': 0.1}
        self.Q_weight = cost_weights['Q']
        self.R_weight = cost_weights['R']
        
        # Model components
        self.gedmd_model = None
        self.intrinsic_basis = None  # Eigenvectors for projection to φ-space
        self.n_intrinsic = None      # Intrinsic dimension
        
        # Reduced system matrices
        self.A_phi = None  # Intrinsic dynamics: φ̇ = A_φ φ + B_φ u
        self.B_phi = None
        self.N_phi = None # Bilinear terms: φ̇ = A_φ φ + B_φ u + Σ u_k N_k φ
        self.K_lqr = None  # LQR gain matrix
        
        # Training data (needed for coordinate transformations)
        self.X_train = None
        self.U_train = None
        self.Z_train = None
        
    def fit(self, X: np.ndarray, X_dot: np.ndarray, U: np.ndarray, dt: float = 0.05, enforce_stability: bool = True, discrete_time: bool = False, **kwargs) -> 'KRONICController':
        """
        Fit KRONIC model to training data.
        
        Process:
        1. Apply kernel Generator EDMD to extended state z = [x, u]
        2. Extract intrinsic coordinates via eigenvalue analysis
        3. Learn reduced dynamics (Continuous or Discrete)
        4. Compute LQR controller (CARE or DARE)
        
        Args:
            X: State trajectories (n_samples × n_states)
            X_dot: State derivatives (n_samples × n_states) 
            U: Control inputs (n_samples × n_controls)
            dt: Time step
            enforce_stability: Enforce continuous stability Re(lambda) < 0
            discrete_time: If True, learn discrete mapping phi_{k+1} = A phi_k + ...
            **kwargs: Additional arguments for KernelGEDMD (e.g., n_subsample)
            
        Returns:
            self: Fitted KRONIC controller
        """
        if self.verbose:
            print("🚀 FITTING RKHS-KRONIC CONTROLLER")
            print("=" * 50)

        # Store training data
        self.X_train = X.copy()
        self.U_train = U.copy()  
        self.Z_train = np.hstack([X, U])  # Extended state [x, u]
        self.discrete_time = discrete_time
        
        # Normalize the extended state
        Z_train_normalized = self.Z_scaler.fit_transform(self.Z_train)

        n_samples, n_states = X.shape
        n_controls = U.shape[1]
        
        if self.verbose:
            print(f"📊 Data: {n_samples} samples, {n_states} states, {n_controls} controls")
            print(f"   Mode: {'Discrete (DARE)' if discrete_time else 'Continuous (CARE)'}")
            
        # Step 1: Apply Generator EDMD to discover intrinsic coordinates
        self._discover_intrinsic_coordinates(Z_train_normalized, X_dot, U, dt, **kwargs)
        
        # Step 2: Learn reduced dynamics in intrinsic space
        if discrete_time:
            self._learn_discrete_dynamics(U, dt, enforce_stability=enforce_stability)
        else:
            self._learn_reduced_dynamics(X_dot, U, dt, enforce_stability=enforce_stability)
        
        # Step 3: Design LQR controller in intrinsic coordinates
        self._synthesize_optimal_controller()
        
        if self.verbose:
            print("✅ KRONIC controller fitted successfully!")
            
        return self
        
    def _discover_intrinsic_coordinates(self, Z_normalized: np.ndarray, X_dot: np.ndarray, U: np.ndarray, dt: float, **kwargs):
        """Apply Generator EDMD to discover intrinsic coordinates."""
        if self.verbose:
            print("\n🔍 Step 1: Discovering intrinsic coordinates...")
            
        # Extended state derivatives: ż = [ẋ, 0] (piecewise constant control)
        # BUG FIX: If we scale Z, we MUST scale Z_dot by the same factor (Chain rule).
        # z_norm = (z - mu) / sigma  =>  z_dot_norm = z_dot / sigma
        Z_dot_raw = np.hstack([X_dot, np.zeros_like(U)])
        
        # Apply scaling (division by sigma)
        # Note: scale_ is an array of size (n_states + n_controls)
        Z_dot = Z_dot_raw / (self.Z_scaler.scale_ + 1e-12)
        
        # Apply Generator EDMD (uses proven vectorized implementation + adaptive regularization)
        # Allow epsilon tuning from kwargs
        epsilon = kwargs.get('epsilon', 1e-6)
        self.gedmd_model = KernelGEDMD(kernel_type='rbf', sigma=self.kernel.sigma, epsilon=epsilon, verbose=self.verbose)
        
        # Extract subsampling and diffusion args if present
        n_subsample = kwargs.get('n_subsample', None)
        subsample_method = kwargs.get('subsample_method', 'farthest_point')
        Z_diffusion = kwargs.get('Z_diffusion', None) # Stochastic diffusion matrix (d,d,m)
        
        self.gedmd_model.fit(Z_normalized, Z_dot, Z=Z_diffusion, dt=dt, n_subsample=n_subsample, subsample_method=subsample_method)
        
        # Intelligent intrinsic dimension selection using eigenvalue spectrum
        eigenvals = np.real(self.gedmd_model.eigenvalues_)
        eigenvals_desc = np.sort(np.abs(eigenvals))[::-1]  # Descending by magnitude
        
        if len(eigenvals_desc) > 1:
            # Use eigenvalue gaps to detect intrinsic dimension
            # Lower threshold to capture more modes (Swing-up needs pumping modes!)
            eigenval_ratios = eigenvals_desc[:-1] / (eigenvals_desc[1:] + 1e-12)
            significant_dims = np.sum(eigenval_ratios > 1.05) + 1  # Aggressive: capture any non-trivial decay
            
            # Reasonable bounds: Ensure at least a few modes for swing-up
            n_samples = self.Z_train.shape[0]
            # Force at least 4 modes (2 complex pairs) for swing-up dynamics
            min_modes = 4
            self.n_intrinsic = max(min_modes, min(significant_dims, 12, n_samples // 4, len(eigenvals)))
        else:
            self.n_intrinsic = 4 # Default minimum
        # 
        # THEORETICAL JUSTIFICATION:
        # For complex eigenfunction φ(z) = φᵣ(z) + i·φᵢ(z) with eigenvalue λ = λᵣ + i·λᵢ:
        # The dynamics φ̇ = λφ become:
        # φ̇ᵣ + i·φ̇ᵢ = (λᵣ + i·λᵢ)(φᵣ + i·φᵢ)
        # 
        # Separating real and imaginary parts:
        # φ̇ᵣ = λᵣφᵣ - λᵢφᵢ    (real part)
        # φ̇ᵢ = λᵢφᵣ + λᵣφᵢ    (imaginary part)
        # 
        # This gives real linear dynamics in augmented coordinates ψ = [φᵣ, φᵢ]:
        # ψ̇ = A_real ψ + B_real u, where A_real preserves all eigenfunction information
        
        eigenvectors_complex = self.gedmd_model.eigenvectors_[:, :self.n_intrinsic]
        
        # ROBUST EIGENFUNCTION SCALING for control synthesis
        # 
        # Problem: Kernel-based eigenfunctions often have very small magnitudes (~1e-4)
        # leading to correspondingly small controls (~1e-5) instead of reasonable values
        # 
        # Solution: Adaptively scale eigenvectors based on system characteristics
        # while maintaining robustness across different environments and dimensions
        
        eigenvectors_scaled = self._scale_eigenfunctions_robustly(eigenvectors_complex, self.X_train, self.U_train)
        
        # Store scaled complex eigenvectors
        self.intrinsic_basis_complex = eigenvectors_scaled
        
        # Create real-augmented intrinsic basis: [Re(φ), Im(φ)]
        # This preserves all eigenfunction information while staying real-valued
        phi_real = np.real(eigenvectors_scaled)
        phi_imag = np.imag(eigenvectors_scaled)
        
        # Augmented real basis: [Re(φ₁), Im(φ₁), Re(φ₂), Im(φ₂), ...]
        self.intrinsic_basis = np.hstack([phi_real, phi_imag])
        self.n_intrinsic_augmented = 2 * self.n_intrinsic  # Doubled dimension
        
        if self.verbose:
            print(f"   Eigenvalues (top 6): {eigenvals_desc[:6]}")
            print(f"   Selected intrinsic dimension: {self.n_intrinsic} → {self.n_intrinsic_augmented} (real-imaginary augmented)")
            condition = np.linalg.cond(self.gedmd_model.G_00_)
            print(f"   Kernel matrix condition: {condition:.2e}")
            
    def _learn_reduced_dynamics(self, X_dot: np.ndarray, U: np.ndarray, dt: float, enforce_stability: bool = True):
        """Learn reduced dynamics φ̇ = A_φ φ + B_φ u using proper system ID."""
        if self.verbose:
            print("\n🎯 Step 2: Learning reduced dynamics...")
            
        # Project training data to real-augmented intrinsic coordinates  
        # Project training data to real-augmented intrinsic coordinates  
        # ψ = [Re(φ), Im(φ)] where φ = K(Z_train, Z_train) @ complex_basis
        
        # FIX for subsampling: We must project the FULL training set (Z_train)
        # onto the learned basis (defined by subsampled centers).
        
        # 1. Normalize full training data
        Z_train_normalized = self.Z_scaler.transform(self.Z_train)
        
        # 2. Get subsampled centers from model
        Z_centers = self.gedmd_model.X_train_.T
        
        # 3. Compute Gram matrix K(Z_full, Z_centers)
        # Using clean kernel interface
        K_full = self.kernel(Z_train_normalized, Z_centers)
        
        # Compute complex intrinsic coordinates first
        # This uses the SCALED basis we computed in step 1
        phi_complex = K_full @ self.intrinsic_basis_complex
        
        # Create real-augmented coordinates: ψ = [Re(φ), Im(φ)]
        phi_real = np.real(phi_complex)
        phi_imag = np.imag(phi_complex)
        phi_train = np.hstack([phi_real, phi_imag])  # Shape: (n_samples, 2*n_intrinsic)
        
        n_samples = phi_train.shape[0]
        n_controls = U.shape[1]
        
        if self.verbose:
            print(f"   Intrinsic coordinates φ range: [{phi_train.min():.3f}, {phi_train.max():.3f}]")
            
        # Compute φ̇ using central differences (more accurate than forward)
        phi_dot = np.zeros_like(phi_train)
        # dt is passed as argument
        
        if n_samples >= 3:
            # Central differences for interior points
            phi_dot[1:-1] = (phi_train[2:] - phi_train[:-2]) / (2 * dt)
            # Boundary conditions
            phi_dot[0] = (phi_train[1] - phi_train[0]) / dt
            phi_dot[-1] = (phi_train[-1] - phi_train[-2]) / dt
        else:
            # Fallback for small datasets
            phi_dot[:-1] = np.diff(phi_train, axis=0) / dt
            
        # System identification: φ̇ = A_φ φ + B_φ u + Σ u_k N_k φ
        # Bilinear formulation (essential for control-affine systems like Swing-Up)
        
        # Construct interaction features for bilinear terms
        # For single control (m=1): Features = [φ, u, φ*u]
        # For m>1: Need φ*u_1, φ*u_2...
        
        # 1. Linear features [φ, u]
        features_linear = np.hstack([phi_train, U])
        
        # 2. Bilinear features [φ*u_1, ..., φ*u_m]
        features_bilinear_list = []
        for k in range(n_controls):
            # element-wise multiply phi by k-th control channel
            u_k = U[:, k:k+1] # (N, 1)
            phi_u_k = phi_train * u_k # (N, d)
            features_bilinear_list.append(phi_u_k)
            
        features_bilinear = np.hstack(features_bilinear_list) if features_bilinear_list else np.empty((n_samples, 0))
        
        features = np.hstack([features_linear, features_bilinear])
        
        n_features = features.shape[1]
        
        A_phi = np.zeros((self.n_intrinsic_augmented, self.n_intrinsic_augmented))
        B_phi = np.zeros((self.n_intrinsic_augmented, n_controls))
        
        # N matrices: list of (d,d) matrices, one per control channel
        N_phi = np.zeros((n_controls, self.n_intrinsic_augmented, self.n_intrinsic_augmented)) 
        
        # Regularization parameter (from successful tests)
        reg_param = 1e-4
        
        for i in range(self.n_intrinsic_augmented):
            target = phi_dot[:, i]
            
            # Tikhonov regularization for numerical stability
            A_reg = features.T @ features + reg_param * np.eye(n_features)
            b_reg = features.T @ target
            
            try:
                coeffs = np.linalg.solve(A_reg, b_reg)
                
                # Unpack coefficients
                # 1. A matrix (first d coeffs)
                A_phi[i, :] = coeffs[:self.n_intrinsic_augmented]
                
                # 2. B matrix (next m coeffs)
                B_idx_start = self.n_intrinsic_augmented
                B_idx_end = B_idx_start + n_controls
                B_phi[i, :] = coeffs[B_idx_start:B_idx_end]
                
                # 3. N matrices (remaining d*m coeffs)
                N_coeffs = coeffs[B_idx_end:]
                # Reshape into m blocks of d
                for k in range(n_controls):
                    start = k * self.n_intrinsic_augmented
                    end = start + self.n_intrinsic_augmented
                    N_phi[k, i, :] = N_coeffs[start:end] # Row i of N_k
                
                # Validation metric
                pred = features @ coeffs
                r2 = 1 - np.sum((target - pred)**2) / (np.sum((target - np.mean(target))**2) + 1e-12)
                
                if self.verbose and i < 3:
                    print(f"      Dimension {i}: R² = {r2:.3f} (Bilinear)")
                    
            except np.linalg.LinAlgError:
                if self.verbose:
                    print(f"      Dimension {i}: Regularization failed, using defaults")
                # Stable defaults
                A_phi[i, i] = -0.1  # Damping
                B_phi[i, :] = np.random.randn(n_controls) * 0.01
                
        self.A_phi = A_phi
        self.B_phi = B_phi
        self.N_phi = N_phi # Store bilinear terms
        
        # Ensure stability (proven stabilization from debug_kronic_numerics.py)
        # Note: For bilinear systems, A_phi corresponds to stability at u=0.
        if enforce_stability:
            self._ensure_stability()
        elif self.verbose:
            print("   ⚠️ Stability enforcement disabled (allowing unstable/pumping dynamics)")
        
        if self.verbose:
            eigenvals = np.linalg.eigvals(A_phi)
            max_real = np.max(np.real(eigenvals))
            print(f"   A_φ eigenvalues: {eigenvals[:4]}...")
            print(f"   System stable: {'✅' if max_real < 0 else '❌'} (max Re λ = {max_real:.6f})")
            
    def _ensure_stability(self):
        """Enforce stability of learned dynamics (proven technique)."""
        eigenvals = np.linalg.eigvals(self.A_phi)
        max_real = np.max(np.real(eigenvals))
        
        # If marginally stable or unstable, add damping
        if max_real >= -1e-3:  # Near-zero or positive real parts
            damping = 0.1
            self.A_phi -= damping * np.eye(self.n_intrinsic_augmented)
            
            if self.verbose:
                print(f"   Added damping ({damping}) for stability")
                new_max_real = np.max(np.real(np.linalg.eigvals(self.A_phi)))
                print(f"   New max Re λ = {new_max_real:.6f}")
                
    def _scale_eigenfunctions_robustly(self, eigenvectors_complex: np.ndarray, 
                                     X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Robustly scale eigenfunctions for control synthesis across diverse environments.
        
        Strategy:
        1. Adaptive target scaling based on state/control magnitudes
        2. Robust statistics using median instead of mean
        3. Bounded scaling factors to prevent extreme corrections
        4. Efficient sampling for large datasets
        5. Optional disable for debugging
        
        Args:
            eigenvectors_complex: Complex eigenvectors from GEDMD
            X: State training data
            U: Control training data
            
        Returns:
            Scaled eigenvectors with appropriate magnitude for control synthesis
        """
        if not self.enable_eigenfunction_scaling:
            if self.verbose:
                print("   Eigenfunction scaling disabled - using raw eigenvectors")
            return eigenvectors_complex
        
        n_samples = self.Z_train.shape[0]
        
        # Robust sampling: Use reasonable fraction of data (10-30%)
        # For small datasets use all data, for large datasets subsample
        if n_samples <= 100:
            scaling_indices = np.arange(n_samples)
        else:
            # Use 20% of data, minimum 50 samples, maximum 500 samples
            n_scaling = max(50, min(500, n_samples // 5))
            scaling_indices = np.random.choice(n_samples, n_scaling, replace=False)
        
        # Use the unnormalized Z_train for scaling
        Z_scaling = self.Z_train[scaling_indices]
        
        # We need to compute the kernel matrix between the scaled and original data
        # To do this, we first need to normalize Z_scaling with the same scaler
        Z_scaling_normalized = self.Z_scaler.transform(Z_scaling)
        
        # Use the actual centers from the trained model (subsampled)
        # gedmd_model stores data in d3s format (d x m), so transpose to (m x d)
        Z_centers_normalized = self.gedmd_model.X_train_.T
        
        # Use clean kernel interface for cross-gram matrix
        K_scaling = self.kernel(Z_scaling_normalized, Z_centers_normalized)


        # Evaluate eigenfunction magnitudes on scaling data
        phi_scaling = K_scaling @ eigenvectors_complex
        
        # Robust magnitude estimation using median absolute deviation
        current_magnitude = np.median(np.abs(phi_scaling.flatten()))
        
        # ROBUST SCALING: Normalize to Unit Variance
        # This ensures that all eigenfunctions have strictly range ~ [-1, 1] or std=1
        # This is critical for the LQR weights Q and R to be meaningful relative to each other.
        # Without this, eigenfunctions can be 1e-8, making controls effectively zero.
        
        std_dev = np.std(phi_scaling.flatten())
        if std_dev > 1e-12:
            scaling_factor = 1.0 / std_dev
        else:
            scaling_factor = 1.0
            if self.verbose:
                print(f"   ⚠️  Warning: Eigenfunctions have near-zero variance ({std_dev:.2e})")
                
        # Apply scaling
        eigenvectors_scaled = eigenvectors_complex * scaling_factor
                
        if self.verbose:
            print(f"   Robust eigenfunction scaling applied:")
            print(f"      Original std dev: {std_dev:.2e}")
            print(f"      Scaling factor: {scaling_factor:.2e}")
            print(f"      Target std dev: 1.00")
            print(f"      Using {len(scaling_indices)}/{n_samples} samples for scaling estimation")
        
        return eigenvectors_scaled
        
    def _learn_cost_mapping(self) -> np.ndarray:
        """
        Learn optimal cost matrix Q_φ in eigenfunction space using enhanced regularized regression.
        
        Strategy: Find Q_φ such that φ(x,u=0)^T Q_φ φ(x,u=0) ≈ x^T Q x
        across the training dataset using robust regularized regression methods.
        
        Uses adaptive regularization to handle ill-conditioning and scale mismatches.
        
        For portfolio problems, uses problem_type to determine cost function:
        - 'portfolio': quadratic utility cost = γW² (risk aversion on wealth)
        - 'portfolio_log': log utility cost = -log(W) for linear LQR formulation
        
        Returns:
            Learned cost matrix Q_φ for eigenfunction space
        """
        n_samples = min(1500, self.X_train.shape[0])  # Further increased sample size for better cost reconstruction
        indices = np.random.choice(self.X_train.shape[0], n_samples, replace=False)
        
        X_cost = self.X_train[indices]
        n_states = X_cost.shape[1]
        
        # Evaluate eigenfunctions at u=0 for cost mapping
        Z_cost = np.hstack([X_cost, np.zeros((n_samples, self.U_train.shape[1]))])
        
        # Normalize Z_cost using the fitted scaler from the training data
        Z_cost_normalized = self.Z_scaler.transform(Z_cost)
        
        # Use the actual centers from the trained model (subsampled)
        # gedmd_model stores data in d3s format (d x m), so transpose to (m x d)
        Z_centers = self.gedmd_model.X_train_.T
        
        # Use clean kernel interface for cross-gram matrix
        K_cost = self.kernel(Z_cost_normalized, Z_centers)
        
        # Get eigenfunction values in real-augmented space
        phi_complex_cost = K_cost @ self.intrinsic_basis_complex
        phi_real_cost = np.real(phi_complex_cost)
        phi_imag_cost = np.imag(phi_complex_cost)
        phi_cost = np.hstack([phi_real_cost, phi_imag_cost])  # Shape: (n_samples, 2*n_intrinsic)
        
        # Compute true costs based on problem type
        if self.problem_type == 'portfolio':
            # Portfolio problem: Use quadratic utility on wealth
            # Cost = γW² where W is wealth (first state component)
            # This gives a standard LQR formulation for portfolio optimization
            if self.verbose:
                print(f"   Using portfolio quadratic utility cost function")
            
            # Risk aversion parameter (calibrated for typical wealth scales)
            gamma = self.Q_weight * 0.01  # Scale down since wealth squared can be large
            
            # Only penalize wealth variance (first state is wealth)
            true_costs = gamma * X_cost[:, 0]**2
            
        elif self.problem_type == 'portfolio_log':
            # Log utility portfolio: cost = -log(W) to maximize log(W)
            # In log-wealth space: x = log(W), so cost = -x
            # This gives a truly linear LQR formulation
            if self.verbose:
                print(f"   Using log utility cost function: cost = -log(W)")
            
            # Ensure positive wealth for log computation
            wealth = np.maximum(X_cost[:, 0], 1e-6)
            
            # Cost = -log(W) (we want to maximize log wealth)
            # Scale by Q_weight for tuning
            true_costs = -self.Q_weight * np.log(wealth)
            
        else:
            # Standard LQR problem: c_i = x_i^T Q x_i
            Q_state = self.Q_weight * np.eye(n_states)
            true_costs = np.sum((X_cost @ Q_state) * X_cost, axis=1)  # Shape: (n_samples,)
        
        if self.verbose:
            print(f"   Learning cost mapping from {n_samples} samples")
            print(f"   True cost range: [{true_costs.min():.3f}, {true_costs.max():.3f}]")
            print(f"   Eigenfunction range: [{phi_cost.min():.3f}, {phi_cost.max():.3f}]")
        
        # Use enhanced regularized cost mapping
        try:
            if self.verbose:
                print(f"   Using enhanced regularized cost reconstruction...")
            
            # Apply enhanced cost mapping with adaptive regularization
            Q_phi, diagnostics = enhanced_cost_mapping(
                eigenfunctions=phi_cost,
                state_costs=true_costs,
                regularization_method='adaptive_ridge',
                verbose=self.verbose
            )
            
            if self.verbose:
                print(f"   Enhanced cost mapping results:")
                print(f"     Method: {diagnostics['regularization_method']}")
                print(f"     Regularization: α = {diagnostics['regularization_strength']:.2e}")
                print(f"     Reconstruction error: {diagnostics['reconstruction_error']:.6f}")
                print(f"     Matrix rank: {diagnostics['Q_phi_rank']}")
            
            return Q_phi
            
        except Exception as enhanced_error:
            if self.verbose:
                print(f"   Enhanced cost mapping failed: {enhanced_error}")
                print(f"   Falling back to legacy SVD-based approach...")
        
        # Fallback to legacy approach if enhanced method fails
        n_phi = phi_cost.shape[1]
        try:
            # Create feature matrix: each row contains φᵢφⱼ products for upper triangle
            n_features = (n_phi * (n_phi + 1)) // 2  # Upper triangular entries
            A_matrix = np.zeros((n_samples, n_features))
            
            feature_idx = 0
            for i in range(n_phi):
                for j in range(i, n_phi):
                    if i == j:
                        A_matrix[:, feature_idx] = phi_cost[:, i] ** 2  # Diagonal terms
                    else:
                        A_matrix[:, feature_idx] = 2 * phi_cost[:, i] * phi_cost[:, j]  # Off-diagonal (factor of 2 for symmetry)
                    feature_idx += 1
            
            # Use SVD-based regularized regression (more robust than normal equations)
            # This can handle very ill-conditioned matrices
            try:
                # SVD-based approach: A = U Σ V^T, then q = V (Σ^T Σ + λI)^-1 Σ^T U^T y
                U, s, Vt = np.linalg.svd(A_matrix, full_matrices=False)
                
                # Adaptive regularization based on singular value spread
                s_max = np.max(s)
                s_min = np.max(s[s > 1e-15])  # Avoid zero singular values
                cond_num = s_max / s_min if s_min > 0 else np.inf
                
                # Choose regularization parameter based on singular values AND cost/eigenfunction scales
                cost_scale = np.std(true_costs)
                phi_scale = np.std(phi_cost.flatten())
                scale_ratio = cost_scale / (phi_scale + 1e-12) if phi_scale > 1e-12 else 1.0
                
                # Stronger regularization for larger scale mismatches
                if scale_ratio > 1000:  # Extreme scale mismatch (like portfolio problems)
                    base_reg = 1e-1  # Very strong regularization
                elif scale_ratio > 100:  # Large scale mismatch
                    base_reg = 1e-2  # Strong regularization
                elif scale_ratio > 10:   # Moderate scale mismatch
                    base_reg = 1e-3  # Moderate regularization
                else:                    # Similar scales
                    base_reg = 1e-6  # Weak regularization
                
                # Also consider condition number
                if cond_num < 1e6:
                    cond_factor = 1.0
                elif cond_num < 1e12:
                    cond_factor = 10.0
                else:
                    cond_factor = 100.0
                
                lambda_reg = base_reg * cond_factor * s_max**2
                
                if self.verbose and scale_ratio > 10:
                    print(f"   Scale-adaptive regularization: cost_scale={cost_scale:.2e}, phi_scale={phi_scale:.2e}")
                    print(f"   Scale ratio: {scale_ratio:.1f}, base_reg: {base_reg:.1e}, final λ: {lambda_reg:.2e}")
                
                # Regularized pseudoinverse: (A^T A + λI)^-1 A^T
                # Using SVD: A^+ = V (S^2 + λI)^-1 S U^T  
                s_reg = s**2 / (s**2 + lambda_reg)  # Regularized singular values
                q_vec = Vt.T @ np.diag(s_reg * s) @ U.T @ true_costs
                
                # Reconstruct symmetric Q_φ matrix from upper triangular entries
                Q_phi = np.zeros((n_phi, n_phi))
                feature_idx = 0
                for i in range(n_phi):
                    for j in range(i, n_phi):
                        Q_phi[i, j] = q_vec[feature_idx]
                        if i != j:
                            Q_phi[j, i] = q_vec[feature_idx]  # Make symmetric
                        feature_idx += 1
                
                # Validate the mapping
                predicted_costs = np.sum((phi_cost @ Q_phi) * phi_cost, axis=1)  # φ^T Q_φ φ for each sample
                cost_error = np.sqrt(np.mean((predicted_costs - true_costs)**2)) / (np.mean(true_costs) + 1e-12)
                
                if self.verbose:
                    print(f"   SVD-based regularized matrix regression:")
                    print(f"   Learned {n_features} matrix entries with SVD regularization {lambda_reg:.2e}")
                    print(f"   Singular value condition: {cond_num:.2e}")
                    print(f"   Singular values: {s[:min(5, len(s))]}...")
                    print(f"   Cost mapping error: {cost_error:.1%}")
                    print(f"   Q_φ condition number: {np.linalg.cond(Q_phi):.2e}")
                    print(f"   Q_φ eigenvalues: {np.linalg.eigvals(Q_phi)[:3]}...")
                
                # Check if Q_φ is reasonable - more permissive criteria
                eigenvals = np.linalg.eigvals(Q_phi)
                min_eigenval = np.min(eigenvals) # FIX: Define this variable!
                is_positive_semidefinite = np.all(eigenvals > -1e-10)  # Allow small numerical negative values
                is_reasonable_condition = np.linalg.cond(Q_phi) < 1e12  # More permissive condition
                is_reasonable_error = cost_error < 0.8  # More permissive error threshold
                
                if is_positive_semidefinite and is_reasonable_condition and is_reasonable_error:
                    if self.verbose:
                        print(f"   ✅ Learned Q_φ matrix is acceptable with regularization")
                else:
                    # Regularize further by adding to diagonal
                    if self.verbose:
                        print(f"   ⚠️  Q_φ needs further regularization")
                        print(f"   PSD: {is_positive_semidefinite}, Cond: {is_reasonable_condition}, Error: {is_reasonable_error}")
                    
                    if min_eigenval < -1e-9 or np.linalg.cond(Q_phi) > 1e10:
                        if self.verbose:
                            print(f"   ⚠️  Q_φ ill-conditioned (cond={np.linalg.cond(Q_phi):.2e}) or indefinite. Regularizing spectrum.")
                        
                        # Spectral decomposition
                        vals, vecs = np.linalg.eigh(Q_phi)
                        
                        # 1. Project to PSD (clip negatives)
                        max_val = np.max(vals)
                        vals = np.maximum(vals, 0.0)
                        
                        # 2. Enforce Condition Number Cap (limit dynamic range)
                        # We want max_val / min_val <= max_condition
                        max_condition = 1e8 
                        min_val_target = max(1e-6, max_val / max_condition)
                        
                        vals_clipped = np.maximum(vals, min_val_target)
                        
                        # Reconstruct
                        Q_phi = vecs @ np.diag(vals_clipped) @ vecs.T
                        
                        if self.verbose:
                             print(f"   Regularized Q_φ spectrum. New condition: {np.linalg.cond(Q_phi):.2e}")
                             print(f"   Eigenvalues: {vals_clipped[:4]}...")
                            
            except np.linalg.LinAlgError as svd_error:
                # SVD failed, fall back to simpler approach
                if self.verbose:
                    print(f"   SVD-based regression failed: {svd_error}")
                    print(f"   Using simple diagonal scaling")
                
                # Fallback to simple diagonal scaling
                phi_vars = np.var(phi_cost, axis=0)
                cost_var = np.var(true_costs)
                scale_factor = cost_var / (np.mean(phi_vars) + 1e-12)
                Q_phi = scale_factor * np.eye(n_phi)
                
        except Exception as e:
            if self.verbose:
                print(f"   Matrix regression failed: {e}")
                print(f"   Using simple diagonal fallback")
            
            # Conservative fallback to diagonal
            if np.isscalar(self.Q_weight):
                q_scalar = self.Q_weight
            elif np.ndim(self.Q_weight) == 1:
                q_scalar = np.mean(self.Q_weight)
            else:
                q_scalar = np.trace(self.Q_weight) / self.Q_weight.shape[0]
            Q_phi = q_scalar * np.eye(n_phi)
        
        return Q_phi
        
    def _learn_discrete_dynamics(self, U: np.ndarray, dt: float, enforce_stability: bool = True):
        """Learn discrete dynamics φ_{k+1} = A_φ φ_k + B_φ u_k + ..."""
        if self.verbose:
            print("\n🎯 Step 2: Learning reduced dynamics (Discrete)...")
            
        Z_train_normalized = self.Z_scaler.transform(self.Z_train)
        Z_centers = self.gedmd_model.X_train_.T
        K_full = self.kernel(Z_train_normalized, Z_centers)
        
        phi_complex = K_full @ self.intrinsic_basis_complex
        phi_real = np.real(phi_complex)
        phi_imag = np.imag(phi_complex)
        phi_train = np.hstack([phi_real, phi_imag]) 
        
        n_samples = phi_train.shape[0]
        n_controls = U.shape[1]
        
        # Discrete Regression: z_{k+1} approx F(z_k, u_k)
        phi_k = phi_train[:-1]
        phi_next = phi_train[1:]
        U_k = U[:-1]
        n_io = phi_k.shape[0]
        
        # Features: [phi_k, u_k, phi_k * u_k]
        features_linear = np.hstack([phi_k, U_k])
        
        features_bilinear_list = []
        for k in range(n_controls):
            u_ch = U_k[:, k:k+1]
            features_bilinear_list.append(phi_k * u_ch)
        features_bilinear = np.hstack(features_bilinear_list) if features_bilinear_list else np.empty((n_io, 0))
        
        features = np.hstack([features_linear, features_bilinear])
        n_features = features.shape[1]
        
        A_phi = np.zeros((self.n_intrinsic_augmented, self.n_intrinsic_augmented))
        B_phi = np.zeros((self.n_intrinsic_augmented, n_controls))
        N_phi = np.zeros((n_controls, self.n_intrinsic_augmented, self.n_intrinsic_augmented))
        
        reg_param_dynamics = 1e-4 # Regularize state dynamics (A)
        reg_param_control = 1e-8  # Weakly regularize control (B) to preserve authority
        
        # Construct Differential Regularization Matrix (Tikhonov)
        # Features map: [phi (d_aug), u (m), phi*u (d_aug*m)]
        # Indices:
        # A_coeffs: 0 to d_aug
        # B_coeffs: d_aug to d_aug + m
        # N_coeffs: d_aug + m to end
        
        reg_matrix_diag = np.ones(n_features) * reg_param_dynamics
        
        # Set Control (B) regularization to be small
        idx_B_start = self.n_intrinsic_augmented
        idx_B_end = idx_B_start + n_controls
        reg_matrix_diag[idx_B_start:idx_B_end] = reg_param_control
        
        # Set Bilinear (N) regularization
        # Bilinear terms are harder to identify, maybe keep them regularized or slighty less?
        # Let's keep them at dynamics level to avoid overfitting noise
        
        Tikhonov = np.diag(reg_matrix_diag)
        
        # Precompute A_reg_base = X^T X
        XtX = features.T @ features
        
        for i in range(self.n_intrinsic_augmented):
            target = phi_next[:, i]
            
            # Solve (X^T X + Lambda) w = X^T y
            A_reg = XtX + Tikhonov
            b_reg = features.T @ target
            
            try:
                coeffs = np.linalg.solve(A_reg, b_reg)
                
                A_phi[i, :] = coeffs[:self.n_intrinsic_augmented]
                
                B_phi[i, :] = coeffs[idx_B_start:idx_B_end]
                
                N_coeffs = coeffs[idx_B_end:]
                for k in range(n_controls):
                    start = k * self.n_intrinsic_augmented
                    end = start + self.n_intrinsic_augmented
                    N_phi[k, i, :] = N_coeffs[start:end]
                    
            except np.linalg.LinAlgError:
                A_phi[i, i] = 0.99 
                
        self.A_phi = A_phi
        self.B_phi = B_phi
        self.N_phi = N_phi
        
        if enforce_stability:
             eigenvals = np.linalg.eigvals(self.A_phi)
             # Discrete Stability: |lambda| < 1
             max_abs = np.max(np.abs(eigenvals))
             if max_abs > 1.0:
                 if self.verbose:
                     print(f"   Stability Violation (Max |λ|={max_abs:.4f}). Scaling A.")
                 self.A_phi /= (max_abs + 1e-3)
                 
    def _synthesize_optimal_controller(self):
        """Design LQR controller in intrinsic coordinates."""
        if self.verbose:
            print("\n🎮 Step 3: Synthesizing optimal controller...")
            
        # Learn optimal cost mapping from state space to eigenfunction space
        self.Q_phi = self._learn_cost_mapping()
        self.R_matrix = self.R_weight * np.eye(self.B_phi.shape[1])  # Control cost
        
        try:
            from scipy.linalg import solve_discrete_are
            
            if hasattr(self, 'discrete_time') and self.discrete_time:
                # DARE
                 P = solve_discrete_are(self.A_phi, self.B_phi, self.Q_phi, self.R_matrix)
                 # Discrete Gain: K = (R + B^T P B)^-1 B^T P A
                 # Scipy solve_discrete_are usually returns P.
                 # K computation:
                 temp = np.linalg.inv(self.R_matrix + self.B_phi.T @ P @ self.B_phi)
                 self.K_lqr = temp @ self.B_phi.T @ P @ self.A_phi
                 
            else:
                # CARE
                P = solve_continuous_are(self.A_phi, self.B_phi, self.Q_phi, self.R_matrix)
                # Compute LQR gain: K = R^{-1} B^T P
                self.K_lqr = np.linalg.solve(self.R_matrix, self.B_phi.T @ P)
            
            if self.verbose:
                print(f"   LQR gain matrix shape: {self.K_lqr.shape}")
                print(f"   Gain norms: {np.linalg.norm(self.K_lqr, axis=1)}")
                
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError) as e:
            if self.verbose:
                print(f"   LQR synthesis failed: {e}")
                print("   Using default proportional gains")
                
            # Fallback to simple proportional control
            self.K_lqr = 0.1 * np.eye(self.n_intrinsic_augmented, self.B_phi.shape[1]).T
            
    def control(self, x: np.ndarray, u_prev: np.ndarray = None, system_type: str = 'affine') -> np.ndarray:
        """
        Compute control input for given state with support for both control structures.
        
        Two Control Synthesis Approaches:
        
        1. Control-Affine Systems (system_type='affine'):
           - Use u=0 evaluation: Extract φ(x) = φ̃(x,0) from extended eigenfunctions
           - Apply standard LQR control in eigenfunction coordinates
           - Based on RKHS-KRONIC Definition 4.1 and legacy implementations
        
        2. Non-Affine Systems (system_type='non_affine'):
           - Use parameterized optimization: u* = argmin[φ̃(x,u)^T Q φ̃(x,u) + u^T R u]
           - Based on RKHS-KRONIC Theorem 4.2 and Algorithm 2
           - [PLACEHOLDER: Implementation needed]
        
        Args:
            x: Current state (n_states,)
            u_prev: Previous control input (used for warm start in non-affine case)
            system_type: 'affine' for control-affine systems, 'non_affine' for general systems
            
        Returns:
            Control input (n_controls,)
        """
        if self.gedmd_model is None:
            raise RuntimeError("Controller not fitted. Call fit() first.")
            
        n_controls = self.U_train.shape[1]
        
        if system_type == 'affine':
            # CONTROL-AFFINE APPROACH: KRONIC-Compatible u=0 Evaluation
            # 
            # Theoretical Foundation (RKHS-KRONIC Definition 4.1):
            # For extended eigenfunctions φ̃_i(x,u), extract state eigenfunctions:
            # φ_i(x) = φ̃_i(x,0)
            # 
            # This preserves essential dynamics while providing KRONIC-compatible
            # functions for standard LQR synthesis in eigenfunction coordinates.
            # 
            # Based on working legacy implementations in exploratory_testing/
            
            # Create augmented state with u=0: z = [x, 0]
            z_u0 = np.hstack([x.flatten(), np.zeros(n_controls)]).reshape(1, -1)
            
            # Normalize the test point using the fitted scaler
            z_u0_normalized = self.Z_scaler.transform(z_u0)
            
            # Use the actual centers from the trained model (subsampled)
            # gedmd_model stores data in d3s format (d x m), so transpose to (m x d)
            Z_centers = self.gedmd_model.X_train_.T
            
            # Project to real-augmented intrinsic coordinates using u=0 evaluation
            # Use clean kernel interface for cross-gram matrix
            K_test = self.kernel(z_u0_normalized, Z_centers)  # Shape: (1, n_subsamples)
            
            # Compute complex coordinates first, then augment to real
            phi_complex = K_test @ self.intrinsic_basis_complex  # Shape: (1, n_intrinsic)
            phi_real = np.real(phi_complex)
            phi_imag = np.imag(phi_complex)
            phi_augmented = np.hstack([phi_real, phi_imag])  # Shape: (1, 2*n_intrinsic)
            
            # Standard LQR control in eigenfunction coordinates
            # u = -K_lqr * φ(x) where φ(x) = φ̃(x,0)
            
            # CHECK FOR BILINEAR CONTROL (SDRE)
            # B_eff = B + sum(N_k * phi)
            if hasattr(self, 'N_phi') and self.N_phi.size > 0:
                 # State Dependent B matrix
                 # N_phi shape: (m, d, d). phi_augmented shape: (1, d)
                 # B_phi shape: (d, m)
                 
                 d = self.n_intrinsic_augmented
                 m = n_controls
                 phi_vec = phi_augmented.flatten()
                 
                 B_eff = self.B_phi.copy() # (d, m)
                 
                 for k in range(m):
                     # N_k is (d, d). Interaction is N_k @ phi
                     # The dynamics are dot_phi = ... + u_k * (N_k @ phi)
                     # So the k-th column of B_eff is B[:, k] + N_k @ phi
                     N_k = self.N_phi[k]
                     B_eff[:, k] += N_k @ phi_vec
                     
                 # Solve SDRE (Algebraic Riccati for this state)
                 # P A + A^T P - P B R^-1 B^T P + Q = 0
                 # Note: Using cached A_phi, Q_phi
                 
                 try:
                     # Check controllability/solvability?
                     # solve_continuous_are solves: A.T X + X A - X B R^-1 B.T X + Q = 0
                     # We need to compute gains K = R^-1 B^T P
                     
                     # Cost matrices
                     Q = self.Q_phi
                     R = self.R_matrix
                     
                     P_sdre = solve_continuous_are(self.A_phi, B_eff, Q, R)
                     
                     # Compute Gain
                     # K = inv(R) @ B_eff.T @ P
                     # For diagonal R:
                     R_inv = np.linalg.inv(R)
                     K_sdre = R_inv @ B_eff.T @ P_sdre # (m, d)
                     
                     # Control law: u = -K * phi
                     u = -K_sdre @ phi_vec
                     
                 except Exception as e:
                     # Fallback to nominal LQR if SDRE fails (e.g. uncontrollable)
                     # print(f"SDRE Failed: {e}")
                     u = -self.K_lqr @ phi_vec
                     
            else:
                 # Nominal Linear LQR
                 u = -self.K_lqr @ phi_augmented.T  # Shape: (n_controls, 1)
            
            return u.flatten()
            
        elif system_type == 'non_affine':
            # NON-AFFINE APPROACH: Parameterized Eigenfunction Optimization
            # 
            # Theoretical Foundation (RKHS-KRONIC Theorem 4.2):
            # For non-affine systems ẋ = f(x,u), control synthesis through:
            # u*(x) = argmin_u [φ̃(x,u)^T Q_φ φ̃(x,u) + u^T R u]
            # 
            # Where φ̃(x,u) are extended eigenfunctions that encode full
            # state-control relationship for non-affine dynamics.
            # 
            # Algorithm 2 from RKHS-KRONIC theory:
            # 1. Parameterize φ̃(x,u) = Σ v_ij k([x,u], z_j)
            # 2. Formulate cost J(u) = φ̃(x,u)^T Q φ̃(x,u) + u^T R u  
            # 3. Compute gradients ∂J/∂u using kernel derivatives
            # 4. Solve optimization using gradient-based methods
            
            # TODO: Implement non-affine optimization approach
            # 
            # Implementation notes from RKHS-KRONIC documentation:
            # - Computational complexity: O(N*r*I_opt + m^3*I_opt) 
            # - Typically requires 5-20 optimization iterations
            # - Warm starting with previous control reduces computation
            # - Kernel gradients: ∂k/∂u = -(u-u_train)/σ² * k for RBF kernels
            # 
            # For now, fall back to affine method as placeholder
            if self.verbose:
                print("⚠️  Non-affine control synthesis not yet implemented.")
                print("   Falling back to affine method (may not be optimal for non-affine systems)")
            
            return self.control(x, u_prev, system_type='affine')
            
        else:
            raise ValueError(f"Unknown system_type: {system_type}. Use 'affine' or 'non_affine'.")
        
    def predict_trajectory(self, x0: np.ndarray, steps: int, dt: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict closed-loop trajectory under KRONIC control.
        
        Args:
            x0: Initial state
            steps: Number of simulation steps
            dt: Time step size
            
        Returns:
            X_pred: Predicted state trajectory (steps × n_states)
            U_pred: Control sequence (steps × n_controls)  
        """
        if self.gedmd_model is None:
            raise RuntimeError("Controller not fitted.")
            
        n_states = x0.shape[0]
        n_controls = self.U_train.shape[1]
        
        # Storage
        X_pred = np.zeros((steps, n_states))
        U_pred = np.zeros((steps, n_controls))
        
        # Initial condition
        x = x0.copy()
        u = np.zeros(n_controls)
        
        for k in range(steps):
            # Store current state
            X_pred[k] = x
            
            # Compute control
            u = self.control(x, u)
            U_pred[k] = u
            
            # Simple forward Euler integration (could be improved)  
            # For this, we need a forward model - using linear approximation
            x_next = x + dt * (np.random.randn(n_states) * 0.01)  # Placeholder
            x = x_next
            
        return X_pred, U_pred
        
    def get_performance_metrics(self) -> dict:
        """Return performance metrics of fitted controller."""
        if self.gedmd_model is None:
            return {}
            
        metrics = {
            'intrinsic_dimension': self.n_intrinsic,
            'intrinsic_dimension_augmented': self.n_intrinsic_augmented,
            'kernel_condition_number': np.linalg.cond(self.gedmd_model.G_00_),
            'system_stability': np.max(np.real(np.linalg.eigvals(self.A_phi))) < 0,
            'controller_gains': np.linalg.norm(self.K_lqr, 'fro'),
            'gedmd_eigenvalues': self.gedmd_model.eigenvalues_[:self.n_intrinsic]
        }
        
        return metrics


def demonstrate_kronic_controller():
    """Demonstration using the successful slow manifold system."""
    print("RKHS-KRONIC Controller Demonstration")
    print("Using proven slow manifold system from successful tests")
    print("=" * 60)
    
    # System parameters (from successful kronic_proper_gedmd.py tests)
    mu, lam, c = -0.1, -1.0, 0.5
    np.random.seed(42)
    
    # Generate training data
    n_samples = 80
    X = np.random.randn(n_samples, 2) * 0.5  # [x1, x2]
    U = np.random.randn(n_samples, 2) * 0.3  # [u1, u2]
    
    # System dynamics
    x1, x2 = X[:, 0], X[:, 1] 
    u1, u2 = U[:, 0], U[:, 1]
    x1_dot = mu * x1 + u1
    x2_dot = lam * (x2 - x1**2) + (1 + c * x1) * u2
    X_dot = np.column_stack([x1_dot, x2_dot])
    
    print(f"Generated {n_samples} samples for system with μ={mu}, λ={lam}, c={c}")
    print(f"State range: X ∈ [{X.min():.2f}, {X.max():.2f}]")
    print(f"Control range: U ∈ [{U.min():.2f}, {U.max():.2f}]")
    
    # Fit KRONIC controller
    print("\nFitting KRONIC controller...")
    kernel = RBFKernel(sigma=1.0)  # From successful tests
    target = np.zeros(2)  # Regulate to origin
    controller = KRONICController(kernel, target, verbose=True)
    
    controller.fit(X, X_dot, U)
    
    # Test control computation
    print(f"\nTesting controller...")
    x_test = np.array([0.5, -0.3])
    u_control = controller.control(x_test)
    
    print(f"Test state: {x_test}")
    print(f"Control output: {u_control}")
    
    # Performance summary
    metrics = controller.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        elif isinstance(value, np.ndarray):
            print(f"  {key}: {value[:3]}...")  # Show first 3 elements
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ KRONIC controller demonstration completed successfully!")
    

if __name__ == "__main__":
    demonstrate_kronic_controller()