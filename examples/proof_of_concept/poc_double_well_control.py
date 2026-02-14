
import numpy as np
import scipy.linalg
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path to import environments
# File: examples/proof_of_concept/poc.py -> Dir: examples/proof_of_concept -> Parent: examples -> Parent: Root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from environments.base_env import BaseEnvironment

# --- 1. Environment Definition ---
class ControlledDoubleWellEnv(BaseEnvironment):
    """
    2D Double Well with Control Input.
    Potential: V(x,y) = (x^2 - 1)^2 + y^2
    Dynamics: dx = -grad(V) + u
    Minima at (-1, 0) and (1, 0). Saddle at (0, 0).
    """
    def __init__(self, dt=0.01, noise_strength=0.0):
        self._dt = dt
        self._state_dim = 2
        self._action_dim = 2
        self.noise_strength = noise_strength
        self.state = self.reset()

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def dt(self):
        return self._dt
        
    def reset(self, state=None):
        if state is None:
            # Start in Left Well (-1, 0) with small noise
            self.state = np.array([-1.0, 0.0]) + np.random.normal(0, 0.05, 2)
        else:
            self.state = np.array(state)
        return self.state
        
    def potential(self, x, y):
        return (x**2 - 1)**2 + y**2
        
    def gradient(self, x, y):
        # dV/dx = 2(x^2 - 1) * 2x = 4x^3 - 4x
        # dV/dy = 2y
        dv_dx = 4 * x**3 - 4 * x
        dv_dy = 2 * y
        return np.array([dv_dx, dv_dy])
        
    def step(self, action):
        u = np.clip(action, -5.0, 5.0) # Limit control authority
        x, y = self.state
        
        grad = self.gradient(x, y)
        
        # Overdamped Langevin dynamics
        # dx = (-gradV + u)dt + sigma * sqrt(dt) * dW
        
        drift_x = (-grad[0] + u[0]) * self.dt
        drift_y = (-grad[1] + u[1]) * self.dt
        
        # Diffusion
        # Klus et al use noise_strength = sigma
        noise = self.noise_strength * np.sqrt(self.dt) * np.random.normal(0, 1, 2)
        
        self.state += np.array([drift_x, drift_y]) + noise
        
        # Reward: Minimize distance to Right Well (1, 0)
        dist = np.linalg.norm(self.state - np.array([1.0, 0.0]))
        reward = -dist # Simple dense reward for logging
        
        done = False
        return self.state, reward, done, {}

# --- 2. Feature Map (Polynomial) ---
from sklearn.preprocessing import PolynomialFeatures

class PolyFeatureMap:
    def __init__(self, degree=3):
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.dim = None
        
    def fit(self, X):
        # X: [N, StateDim]
        self.poly.fit(X)
        self.dim = self.poly.n_output_features_
        print(f"Polynomial Features (Degree 3) Dim: {self.dim}")
        
    def transform(self, x):
        # x: [Batch, StateDim] or [StateDim]
        is_batch = x.ndim > 1
        if not is_batch:
            x = x.reshape(1, -1)
            
        feat = self.poly.transform(x)
        
        if is_batch:
            return feat # [N, Dim]
        else:
            return feat.flatten() # [Dim]
            
    def get_derivatives(self, x):
        """
        Compute gradient (J) and Laplacian (L) of features at x.
        x: [StateDim]
        Returns:
            J: [Dim, StateDim] (Jacobian of phi w.r.t x)
            L: [Dim] (Laplacian of phi)
        """
        # Manual implementation for Degree 3 Polynomials in 2D
        # Features: [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3] (approx order)
        # Note: Scikit-learn ordering is slightly different, we need to be careful.
        # Ideally we would auto-diff this, but for PoC we can use finite differences
        # or just rely on the fact that for Polynomials we can compute exact derivatives.
        
        # Finite Difference Fallback (Robust and Easy)
        eps = 1e-4
        n_dim = x.shape[0]
        n_feat = self.dim
        J = np.zeros((n_feat, n_dim))
        L = np.zeros(n_feat)
        
        phi_0 = self.transform(x)
        
        # Forward/Backward for Gradient and Laplacian
        for i in range(n_dim):
            x_plus = x.copy(); x_plus[i] += eps
            x_minus = x.copy(); x_minus[i] -= eps
            
            phi_plus = self.transform(x_plus)
            phi_minus = self.transform(x_minus)
            
            # Central Difference for Gradient: df/dx ~ (f(x+h) - f(x-h)) / 2h
            J[:, i] = (phi_plus - phi_minus) / (2 * eps)
            
            # Central Difference for Laplacian: d2f/dx2 ~ (f(x+h) - 2f(x) + f(x-h)) / h^2
            d2_phi = (phi_plus - 2*phi_0 + phi_minus) / (eps**2)
            L += d2_phi # Sum of second derivatives
            
        return J, L
        
    def get_dim(self):
        return self.dim

# --- 3. Koopman Dynamics Learner (Bilinear) ---
class KoopmanLearner:
    def __init__(self, feature_map, state_dim=2, action_dim=2):
        self.phi = feature_map
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def fit(self, X, U, Xn, noise_strength=0.0):
        # Extended Bilinear EDMD
        # z_{t+1} = A z_t + B u_t + sum_i u_i B_i z_t
        # Regressors: [z, u, z*u_1, z*u_2]
        
        Z = self.phi.poly.transform(X) # [N, Z_dim]
        Zn = self.phi.poly.transform(Xn)
        self.z_dim = Z.shape[1]
        
        # Construct Feature Matrix G
        # G = [Z, U, Z*u1, Z*u2]
        G_parts = [Z, U]
        for i in range(self.action_dim):
            u_i = U[:, i:i+1] # [N, 1]
            G_parts.append(Z * u_i) # [N, Z_dim]
            
        G = np.hstack(G_parts) 
        Y = Zn
        
        # Ridge Regression
        lam = 1e-4
        GT_G = G.T @ G
        GT_Y = G.T @ Y
        
        K_T = scipy.linalg.solve(GT_G + lam * np.eye(G.shape[1]), GT_Y)
        K = K_T.T # [Z_dim, Total_Input_Dim]
        
        # Parse K
        # Indices:
        # A: 0 to z_dim
        # B_lin: z_dim to z_dim + action_dim
        # B_bilin: rest
        
        idx = 0
        self.A = K[:, idx : idx + self.z_dim]; idx += self.z_dim
        self.B_lin = K[:, idx : idx + self.action_dim]; idx += self.action_dim
        
        self.B_tensor = np.zeros((self.action_dim, self.z_dim, self.z_dim))
        for i in range(self.action_dim):
            self.B_tensor[i] = K[:, idx : idx + self.z_dim]
            idx += self.z_dim
            
        evals = np.linalg.eigvals(self.A)
        print(f"Learned Extended Bilinear. A: {self.A.shape}, B_lin: {self.B_lin.shape}")
        print(f"Max A Eigenvalue: {np.max(np.abs(evals)):.4f}")
        
    def predict(self, z, u):
        # z: [Z_dim], u: [ActionDim]
        # z_next = A z + B_lin u + sum B_i z u_i
        res = self.A @ z + self.B_lin @ u
        for i in range(self.action_dim):
             res += self.B_tensor[i] @ z * u[i]
        return res

    def get_B_matrix(self, z):
        # B(z) u = B_lin u + sum u_i B_i z
        # B(z)[:, i] = B_lin[:, i] + B_tensor[i] @ z
        
        B_z = np.zeros((self.z_dim, self.action_dim))
        for i in range(self.action_dim):
            B_z[:, i] = self.B_lin[:, i] + self.B_tensor[i] @ z
        return B_z

# --- 4. SDRE Solver (State Dependent LQR) ---
def solve_lqr_sdre(A, B_z, Q, R):
    # Solves DARE for specific pair (A, B(z))
    # Note: A implies the drift dynamics.
    # In Bilinear: z_{k+1} = A z_k + B(z_k) u_k
    # This fits the LQR form x_{k+1} = A x_k + B u_k
    
    try:
        P = scipy.linalg.solve_discrete_are(A, B_z, Q, R)
        inv_term = R + B_z.T @ P @ B_z
        K = np.linalg.solve(inv_term, B_z.T @ P @ A)
        return K
    except Exception as e:
        # Fallback for unstabilizable pair (e.g. at fixed points where B=0?)
        # print(f"SDRE Fail: {e}")
        return np.zeros((B_z.shape[1], A.shape[0]))


# --- 5. Main Experiment ---
def run_double_well_control():
    print("--- Prototyping Double Well Control (KRONIC) ---")
    
    # Init Env and Features
    env = ControlledDoubleWellEnv() # High noise for testing
    learning_env = ControlledDoubleWellEnv(noise_strength=0.01) # Low noise for learning
    feats = PolyFeatureMap(degree=3)
    
    # 1. Data Collection (Global Low Noise)
    print("Collecting Global Low Noise Data...")
    X, U, Xn = [], [], []
    
    # Global Uniform Sampling (Ensures coverage)
    for _ in range(5000):
        state = np.random.uniform(-2, 2, size=(2,))
        learning_env.state = state.copy() # Safety
        u = np.random.uniform(-3, 3, size=(2,))
        
        # Save PRE-step state
        X.append(state.copy())
        U.append(u.copy())
        
        next_state, _, _, _ = learning_env.step(u)
        Xn.append(next_state.copy())
        
    # Saddle Targeted
    print("Collecting Saddle Data...")
    for _ in range(1000):
        state = np.random.normal(0, 0.2, size=(2,))
        learning_env.state = state
        u = np.random.uniform(-2, 2, size=(2,))
        next_state, _, _, _ = learning_env.step(u)
        
        X.append(state)
        U.append(u)
        Xn.append(next_state)

    X = np.array(X)
    U = np.array(U)
    Xn = np.array(Xn)
    
    print(f"Data Stats: X_min={X.min(axis=0)}, X_max={X.max(axis=0)}")
    
    # Fit Features
    feats.fit(X)
    
    # 2. Learn Koopman (Stochastic gEDMD)
    learner = KoopmanLearner(feats)
    # We pass the noise strength so the learner knows the physics context if needed
    # (Currently implicitly handled by finite difference on noisy data)
    learner.fit(X, U, Xn, noise_strength=0.5)
    
    # --- Validation: Check Vector Field Fit ---
    print("Validating Learned Dynamics...")
    val_x = np.linspace(-1.5, 1.5, 50)
    val_X = np.column_stack([val_x, np.zeros_like(val_x)]) # y=0 section
    val_U = np.zeros((50, 2)) # u=0
    
    # True Vector Field (dx = 4x - 4x^3)
    true_dx = 4 * (val_x - val_x**3) * env.dt # Discrete step
    
    # Predicted Vector Field
    val_Z = feats.transform(val_X)
    val_Zn = learner.predict(val_Z.T, val_U.T).T # [50, Z_dim]
    
    # To get predicted x_next, we need to invert Z -> x?
    # Or just look at the first dimension of Z (since Poly degree 1 is x)
    # feats.poly output: [x, y, x^2, xy, y^2, ...]
    # So Z[:, 0] should be x (if include_bias=False)
    # Let's verify feature names
    print(f"Feature Names: {feats.poly.get_feature_names_out(['x', 'y'])}")
    
    pred_xn = val_Zn[:, 0] # indices depend on ordering...
    # Reconstructing x from Z?
    # We can just learn a projection C: Z -> x for verification
    C_val = np.linalg.lstsq(feats.transform(X), X, rcond=None)[0].T
    # pred_Xn_rec = val_Zn @ C_val.T
    # Actually, simpler: just assume first col is x for now or verify 
    
    # Let's verify ordering exactly
    names = feats.poly.get_feature_names_out(['x', 'y'])
    x_idx = np.where(names == 'x')[0][0]
    
    pred_xn = val_Zn[:, x_idx]
    pred_dx = pred_xn - val_x
    
    mse = np.mean((pred_dx - true_dx)**2)
    null_mse = np.mean((0 - true_dx)**2)
    print(f"Dynamics MSE (x-axis): {mse:.6f}")
    print(f"Null Model MSE       : {null_mse:.6f}")
    
    # Plot Validation
    plt.figure()
    plt.plot(val_x, true_dx/env.dt, 'k--', label='True dx/dt')
    plt.plot(val_x, pred_dx/env.dt, 'r-', label='Pred dx/dt')
    plt.title("Vector Field Validation (y=0, u=0)")
    plt.xlabel("x")
    plt.ylabel("dx/dt")
    plt.legend()
    plt.grid(True)
    plt.savefig('double_well_validation.png')
    print("Saved double_well_validation.png")
    
    # 3. Design Controller (Target Origin / Saddle)
    print("Designing SDRE for Saddle Point Stabilization (Origin)...")
    # Target: Origin (0, 0)
    target_state = np.array([0.0, 0.0])
    z_target = feats.transform(target_state)
    
    Q = np.eye(learner.z_dim) * 50.0 # Intermediate State Penalty
    R = np.eye(learner.action_dim) * 0.1 # Moderate Control
    
    # 4. Test "Saddle Stabilization" (Multi-Start)
    print("\n--- Testing Saddle Stabilization (Multiple Starts -> Origin) ---")
    start_states = [
        [-1.0, 0.0], # Left Well (Stable)
        [1.0, 0.0],  # Right Well (Stable)
        [0.0, 1.0],  # High Potential
        [-0.5, -0.5] # Intermediate
    ]
    
    plt.figure(figsize=(12, 5))
    
    # Phase Space
    plt.subplot(1, 2, 1)
    gx = np.linspace(-2, 2, 100)
    gy = np.linspace(-1, 1, 100)
    GX, GY = np.meshgrid(gx, gy)
    V = (GX**2 - 1)**2 + GY**2
    plt.contourf(GX, GY, V, levels=20, cmap='viridis', alpha=0.3)
    plt.title("Phase Space Trajectories")
    plt.xlabel("x"); plt.ylabel("y")
    
    # Time Series
    plt.subplot(1, 2, 2)
    plt.title("Time Series (x coordinate)")
    plt.xlabel("Step"); plt.ylabel("x")
    plt.axhline(0.0, color='k', linestyle='--', label='Target')

    colors = ['r', 'b', 'm', 'c']
    
    for i, start_s in enumerate(start_states):
        state = env.reset(state=start_s)
        # Fix: Copy state to avoid reference aliasing
        trajectory = [state.copy()]
        
        # 1000 steps = 50s (dt=0.05). 
        # Should stabilize much faster (tau ~ 0.25s)
        for t in range(1000): 
            z = feats.transform(state)
            z_err = z - z_target
            
            # SDRE Control
            B_z = learner.get_B_matrix(z)
            K_sdre = solve_lqr_sdre(learner.A, B_z, Q, R)
            
            u = -K_sdre @ z_err
            u = np.clip(u, -5, 5)
            
            next_state, _, _, _ = env.step(u)
            state = next_state
            trajectory.append(state.copy()) # Fix: Copy
            
        traj = np.array(trajectory)
        print(f"Start {start_s} -> Final {traj[-1]}")
        
        # Plot Phase
        plt.subplot(1, 2, 1)
        # Use linewidth 2 for visibility
        plt.plot(traj[:, 0], traj[:, 1], color=colors[i], marker='.', markevery=100, linewidth=2, alpha=0.8, label=f"Start {start_s}")
        plt.scatter(start_s[0], start_s[1], color=colors[i], marker='x', s=100, linewidth=2)
        
        # Plot Time
        plt.subplot(1, 2, 2)
        plt.plot(traj[:, 0], color=colors[i], linewidth=2, label=f"Start {start_s}")

    plt.subplot(1, 2, 1); plt.legend(); plt.scatter([0], [0], color='green', marker='*', s=150, zorder=10)
    plt.subplot(1, 2, 2); plt.legend()
    
    plt.tight_layout()
    plt.savefig('double_well_saddle_multi.png', dpi=150)
    print("Saved double_well_saddle_multi.png")
    
    # 5. Stochastic Test (Robustness) - Multi-Start
    print("\n--- Testing Stochastic Saddle Stabilization (Multi-Start, Noise = 0.5) ---")
    
    # Use explicit axes to prevent clearing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Phase Space (ax1)
    c_plot = ax1.contourf(GX, GY, V, levels=30, cmap='magma', alpha=0.3)
    fig.colorbar(c_plot, ax=ax1, label='Potential')
    ax1.set_title(f"Stochastic Stabilization ($\sigma$=0.5)")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.set_xlim(-2.0, 2.0)
    ax1.set_ylim(-1.5, 1.5)
    
    # Time Series (ax2)
    ax2.set_title("Stochastic Time Series (x)")
    ax2.set_xlabel("Step"); ax2.set_ylabel("x")
    ax2.axhline(0.0, color='k', linestyle='--', label='Target')
    
    stoch_env = ControlledDoubleWellEnv(dt=0.05, noise_strength=0.5)
    
    stats_msg = []
    
    for i, start_s in enumerate(start_states):
        stoch_state = stoch_env.reset(state=start_s)
        # Fix: Copy state
        stoch_traj = [stoch_state.copy()]
        actions = []
        
        steps = 1000
        for t in range(steps):
            z = feats.transform(stoch_state)
            
            # SDRE Control
            B_z = learner.get_B_matrix(z)
            K_sdre = solve_lqr_sdre(learner.A, B_z, Q, R)
            
            u = -K_sdre @ (z - z_target)
            u = np.clip(u, -5, 5)
            
            next_state, _, _, _ = stoch_env.step(u)
            stoch_state = next_state
            stoch_traj.append(stoch_state.copy()) # Fix: Copy
            actions.append(u)

        stoch_traj = np.array(stoch_traj)
        final_err = np.linalg.norm(stoch_traj[-1] - target_state)
        
        # Stats
        std_x = np.std(stoch_traj[:, 0])
        msg = f"Start {start_s}: Final Err={final_err:.4f}, Std(x)={std_x:.4f}"
        print(msg)
        stats_msg.append(msg)
        
        # Plot Phase
        c = colors[i]
        ax1.plot(stoch_traj[:, 0], stoch_traj[:, 1], color=c, alpha=0.8, linewidth=1.5, label=f"Start {start_s}") 
        ax1.scatter(start_s[0], start_s[1], color=c, marker='x', s=100, linewidth=2)
        ax1.scatter(stoch_traj[-1, 0], stoch_traj[-1, 1], color=c, marker='o', s=60, edgecolors='k') 
        
        # Plot Time
        ax2.plot(stoch_traj[:, 0], color=c, alpha=0.8, linewidth=1.5, label=f"Start {start_s}")

    ax1.scatter([0], [0], color='lime', marker='*', s=200, edgecolors='k', zorder=10, label='Target')
    ax1.legend(loc='lower right', fontsize='small', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    ax2.legend(fontsize='small', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('double_well_stochastic_multi.png', dpi=150)
    print("Saved double_well_stochastic_multi.png")
    
    # 6. High-Frequency Control comparison
    print("\n--- High Frequency Control Analysis ---")
    print(f"Standard (20Hz, dt=0.05) Std Dev: {[s.split('Std(x)=')[1] for s in stats_msg]}")
    
    # Hypothesis: Faster actuation reduces stochastic variance
    
    # 1. Discretize Learned Generator at dt=0.01
    dt_fast = 0.01
    A_fast = np.eye(learner.z_dim) + learner.Ac * dt_fast
    # Simplified B discretization
    B_fast_base = (np.eye(learner.z_dim) + 0.5 * learner.Ac * dt_fast) @ learner.Bc * dt_fast
    
    stoch_env_fast = ControlledDoubleWellEnv(dt=dt_fast, noise_strength=0.5)
    state = stoch_env_fast.reset(state=[-0.5, 0.5]) # Start in basin
    traj_fast = [state.copy()]
    
    # Run for same WALL TIME (50s) -> 5000 steps
    steps_fast = 5000
    
    for t in range(steps_fast):
        z = feats.transform(state)
        # Control Law (Use base B matrix for simplicity/speed or recompute)
        # Ideally using SDRE with A_fast, B_fast_base is enough for LQR check
        # But for full fairness let's use fixed gain LQR at Origin to speed up
        # or SDRE if fast enough. 5000 steps SDRE might be slow.
        # Let's use Fixed Gain LQR at origin for HF comparison to show bandwidth effect
        # Or just SDRE. Code below uses Fixed Gain implicitly (solve_lqr_sdre at every step)
        
        K_fast = solve_lqr_sdre(A_fast, B_fast_base, Q, R)
        u = -K_fast @ (z - z_target)
        u = np.clip(u, -5, 5)
        
        next_s, _, _, _ = stoch_env_fast.step(u)
        state = next_s
        traj_fast.append(state.copy())
        
    traj_fast = np.array(traj_fast)
    std_fast = np.std(traj_fast[-1000:], axis=0) # Last 10s
    print(f"High Freq (100Hz) Std Dev: {std_fast}")
    
    # Plot Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot LF (Re-use first multi-start traj which is representative)
    # Just need one LF example.
    # We can't easily access 'stoch_traj' from loop. 
    # Let's re-run a quick LF traj or just trust the print?
    # User wants plot.
    
    stoch_env_lf = ControlledDoubleWellEnv(dt=0.05, noise_strength=0.5)
    s_lf = stoch_env_lf.reset(state=[-0.5, 0.5])
    traj_lf = [s_lf.copy()]
    for _ in range(1000):
        z = feats.transform(s_lf)
        B_z = learner.get_B_matrix(z)
        K_sdre = solve_lqr_sdre(learner.A, B_z, Q, R)
        u = -K_sdre @ (z - z_target)
        u = np.clip(u, -5, 5)
        s_lf = stoch_env_lf.step(u)[0]
        traj_lf.append(s_lf.copy())
    traj_lf = np.array(traj_lf)
    
    # Plot LF
    ax1.set_title("Low Freq Control (20Hz, dt=0.05)")
    ax1.plot(traj_lf[:, 0], color='r', alpha=0.8, linewidth=1, label='x(t)')
    ax1.axhline(0.0, color='k', linestyle='--')
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_xlabel("Steps (dt=0.05)")
    ax1.legend()
    
    # Plot HF
    ax2.set_title("High Freq Control (100Hz, dt=0.01)")
    ax2.plot(traj_fast[:, 0], color='b', alpha=0.8, linewidth=1, label='x(t)')
    ax2.axhline(0.0, color='k', linestyle='--')
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlabel("Steps (dt=0.01)")
    ax2.legend()
    
    plt.suptitle(f"Variance Reduction via Bandwidth ($\sigma=0.5$)\nLF Std: {np.std(traj_lf[:,0]):.3f} -> HF Std: {np.std(traj_fast[:,0]):.3f}")
    plt.tight_layout()
    plt.savefig('double_well_high_freq.png', dpi=150)
    print("Saved double_well_high_freq.png")
