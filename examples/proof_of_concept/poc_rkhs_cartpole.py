
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import sys
import os

# Set backend to Agg to avoid display errors
import matplotlib
matplotlib.use('Agg')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.kernels import PeriodicKernel, GibbsKernel
from environments.cartpole_env import CartPoleEnv
from matplotlib.animation import FuncAnimation, PillowWriter

def save_animation(traj, filename='swingup.gif', dt=0.02):
    """
    Generate an animation of the CartPole trajectory.
    traj: (T, 4) state trajectory [x, xd, th, thd]
    """
    print(f"Generating animation {filename}...")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    # ax.set_aspect('equal') # Disable to prevent fighting with dynamic xlim
    ax.grid(True)
    
    cart_width = 0.5
    cart_height = 0.3
    pole_len = 1.0 # Visual length
    
    cart_rect = plt.Rectangle((0, 0), cart_width, cart_height, fc='black')
    pole_line, = ax.plot([], [], 'r-', linewidth=3)
    ax.add_patch(cart_rect)
    
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
    
    def init():
        cart_rect.set_xy((-cart_width/2, -cart_height/2))
        pole_line.set_data([], [])
        time_text.set_text('')
        return cart_rect, pole_line, time_text
        
    def update_frame(frame):
        state = traj[frame]
        x = state[0]
        theta = state[2] 
        
        # Camera Tracking
        ax.set_xlim(x - 5.0, x + 5.0)
        
        # Cart pos centered at x
        cart_rect.set_xy((x - cart_width/2, -cart_height/2))
        
        # Pole tip
        x_pole = x + pole_len * np.sin(theta)
        y_pole = pole_len * np.cos(theta)
        
        pole_line.set_data([x, x_pole], [0, y_pole])
        time_text.set_text(f'Time: {frame*dt:.2f}s')
        
    # Manual save loop to ensure limits are respected
    writer = PillowWriter(fps=30)
    with writer.saving(fig, filename, dpi=80):
        # Subsample if too many frames (max 600 frames for 20s at 30fps)
        # traj length 2000. 30fps -> 33ms. dt=0.005. 0.033/0.005 approx 6 steps.
        step = max(1, int(len(traj) / 300)) # Aim for ~300 frames GIF
        for i in range(0, len(traj), step):
            update_frame(i)
            writer.grab_frame()
            if i % 100 == 0:
                print(f"Rendered frame {i}/{len(traj)}. x={traj[i,0]:.2f} xlim={ax.get_xlim()}")
                
    plt.close(fig)
    print(f"Animation saved to {filename}")

def main():
    np.random.seed(42)
    print("Running RKHS-KRONIC CartPole Control (Staged A/B/N Learning)...")
    
    # ============================================
    # STAGED DATA COLLECTION (from staged_koopman_control_theory.md)
    # Stage 1: Autonomous (u=0) -> Learn A
    # Stage 2: Equilibrium (x≈0) with control -> Learn B  
    # Stage 3: Full controlled -> Learn N
    # ============================================
    
    env = CartPoleEnv()
    lag_steps = 4  # Effective dt = 0.02s
    dt = env.dt * lag_steps
    
    print(f"\n=== STAGED DATA COLLECTION (dt={dt:.3f}s) ===")
    
    # STAGE 1: Autonomous Data (u=0) for learning A
    N_auto = 3000
    print(f"\nStage 1: Collecting {N_auto} autonomous samples (u=0)...")
    X_auto = np.zeros((N_auto, 4))
    Y_auto = np.zeros((N_auto, 4))
    
    # ANCHOR: Force (0,0) -> 0
    # Add copies of the exact equilibrium to PIN the model there
    n_anchor = 200
    X_auto[:n_anchor] = 0.0
    Y_auto[:n_anchor] = 0.0 # Autonomous fixed point
    
    for i in range(n_anchor, N_auto):
        # Cover full state space
        state = np.random.uniform(low=[-3, -3, -np.pi, -5], high=[3, 3, np.pi, 5])
        X_auto[i] = state
        env.state = state.copy()
        
        for _ in range(lag_steps):
            next_state, _, _, _ = env.step(0.0)  # u = 0
        Y_auto[i] = next_state
    
    # STAGE 2: Equilibrium Data (x≈0) with varying control for learning B
    N_eq = 2000
    print(f"Stage 2: Collecting {N_eq} equilibrium samples (x≈0, varying u)...")
    X_eq = np.zeros((N_eq, 4))
    U_eq = np.zeros((N_eq, 1))
    Y_eq = np.zeros((N_eq, 4))
    
    # Add anchors for B learning too? 
    # At x=0, z_next - A z = B u. If u=0, residual=0.
    # Our regression (residual = B u) should handle u=0 -> residual=0 naturally (no intercept).
    
    for i in range(N_eq):
        # Near upright equilibrium (small perturbations)
        state = np.random.normal(scale=[0.3, 0.3, 0.15, 0.3])
        X_eq[i] = state
        u = np.random.uniform(-15, 15)  # Varying control
        U_eq[i] = u
        
        env.state = state.copy()
        for _ in range(lag_steps):
            next_state, _, _, _ = env.step(u)
        Y_eq[i] = next_state
    
    # STAGE 3: Full Controlled Data for learning N (bilinear interaction)
    N_ctrl = 5000
    print(f"Stage 3: Collecting {N_ctrl} full controlled samples...")
    X_ctrl = np.zeros((N_ctrl, 4))
    U_ctrl = np.zeros((N_ctrl, 1))
    Y_ctrl = np.zeros((N_ctrl, 4))
    
    for i in range(N_ctrl):
        # Full state space with control
        state = np.random.uniform(low=[-5, -5, -np.pi, -5], high=[5, 5, np.pi, 5])
        X_ctrl[i] = state
        u = np.random.uniform(-15, 15)
        U_ctrl[i] = u
        
        env.state = state.copy()
        for _ in range(lag_steps):
            next_state, _, _, _ = env.step(u)
        Y_ctrl[i] = next_state
    
    print(f"Total samples: {N_auto + N_eq + N_ctrl}")
    
    # Combine for kernel centering (use all data for shared feature space)
    X_all = np.vstack([X_auto, X_eq, X_ctrl])
    N_total = X_all.shape[0]
        
    
    # 2. Periodic Kernel (Stationary Baseline)
    print("Initializing Periodic Kernel (l=4.0)...")
    # Kernel operates on [xd, th, thd] (3 dims). Theta is at index 1 in this reduced space.
    kernel = PeriodicKernel(period=2*np.pi, length_scale=4.0, periodic_dims=[1])
    
    # 2b. Nystrom Features
    n_centers = 1500
    # Select centers randomly from ALL data (shared feature space)
    inds = np.random.choice(N_total, n_centers, replace=False)
    centers_raw = X_all[inds]
    centers = centers_raw[:, 1:]  # Drop x, keep [xd, th, thd]
    # centers_raw no longer needed for kernel, but maybe for other things? 
    # Logic: Centers used for kernel(centers, centers). Must match X's dim.
    
    # Feature Map
    # Pre-compute Kernel Matrix on Centers (for Nystrom normalization if needed, or just RFF)
    # We use Empirical Kernel Map: phi(x) = k(x, c_i)
    # Ideally should orthonormalize. Here just raw kernel features.
    
    # Orthonormalization (PCA/SVD on feature space)
    print(f"Computing Kernel Matrix (N={N_total}, M={n_centers})...")
    
    # Calculate K_MM (Centers vs Centers)
    K_mm = kernel(centers, centers) 
    K_mm += 1e-6 * np.eye(n_centers) # Regularize

    feat_map_mat = np.eye(n_centers) # Placeholder if we don't whiten
    
    # Compute Eigenfunctions of the Kernel (EDMD style)
    eigvals, eigvecs = scipy.linalg.eigh(K_mm)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Keep top components 
    n_features = 300 
    D_inv_sqrt = np.linalg.inv(np.diag(np.sqrt(eigvals[:n_features])))
    feat_map_mat = D_inv_sqrt @ eigvecs[:, :n_features].T
    
    print(f"Retained Top {n_features} Eigenfeatures.")

    # STRICT CENTERING: Enforce phi(0) = 0
    # KERNEL PART REF
    # [xd, th=0, thd] -> [0, 0, 0]
    target_k_in = np.array([0., 0., 0.]).reshape(1, 3) 
    k_cx_ref = kernel(centers, target_k_in)
    z_k_ref = (feat_map_mat @ k_cx_ref).T # (1, n_features)
    
    # STATE PART REF
    # [x, xd, cos(0)=1, sin(0)=0, thd] -> [0, 0, 1, 0, 0]
    z_s_ref = np.array([0., 0., 1., 0., 0.]).reshape(1, 5)
    
    print(f"Centering Features: Subtracting Reference z_ref (norm={np.linalg.norm(np.hstack([z_k_ref, z_s_ref])):.4f})")
    
    # Feature Map Function
    def get_features(X_in):
        # Ignore x (column 0) for the kernel part
        # X_in = [x, xd, th, thd] -> X_reduced = [xd, th, thd]
        X_reduced = X_in[:, 1:]  # Drop x
        k_cx = kernel(centers, X_reduced)
        z_k = (feat_map_mat @ k_cx).T
        
        # Center Kernel Features
        z_k = z_k - z_k_ref
        
        # PRINCIPLED: Vanilla state augmentation only
        # X_in = [x, xd, th, thd]
        x_aug = X_in[:, 0:1]   # x
        xd_aug = X_in[:, 1:2]  # xd
        th = X_in[:, 2:3]
        c_th = np.cos(th)
        s_th = np.sin(th)
        thd_aug = X_in[:, 3:4] # thd
        
        # Clean Augmentation: [x, xd, cos(th), sin(th), thd]
        X_trig = np.hstack([x_aug, xd_aug, c_th, s_th, thd_aug])
        
        # Center State Features
        X_trig = X_trig - z_s_ref
        
        return np.hstack([z_k, X_trig])

    print("\nExtracting Features for each stage...")
    
    # Get features for each dataset
    Z_auto = get_features(X_auto)
    Z_auto_next = get_features(Y_auto)
    
    Z_eq = get_features(X_eq)
    Z_eq_next = get_features(Y_eq)
    
    Z_ctrl = get_features(X_ctrl)
    Z_ctrl_next = get_features(Y_ctrl)
    
    r = Z_auto.shape[1]
    print(f"Feature Dimension r={r} (Kernel={n_features} + State=5)")
    
    # ============================================
    # STAGED LEARNING (from staged_koopman_control_theory.md)
    # ============================================
    
    print("\n=== STAGE 1: Learning A from Autonomous Data ===")
    # Z_next = A_d @ Z  (discrete time)
    # Regression: minimize ||Z_next - A_d @ Z||^2
    alpha = 1e-6
    lhs_A = Z_auto.T @ Z_auto + alpha * np.eye(r)
    rhs_A = Z_auto.T @ Z_auto_next
    A_discrete = np.linalg.solve(lhs_A, rhs_A).T  # (r, r)
    
    # Validate A on autonomous data
    Z_auto_pred = Z_auto @ A_discrete.T
    SS_res_A = np.sum((Z_auto_next - Z_auto_pred)**2)
    SS_tot_A = np.sum((Z_auto_next - Z_auto_next.mean(0))**2)
    R2_A = 1 - SS_res_A / SS_tot_A
    print(f"   Stage 1 R² (A only, autonomous): {R2_A:.4f}")
    
    # Convert to continuous
    A_cont = (A_discrete - np.eye(r)) / dt
    eigs_A = np.linalg.eigvals(A_cont)
    n_unstable = np.sum(eigs_A.real > 0)
    print(f"   A eigenvalues: {n_unstable} unstable, {r - n_unstable} stable")
    print(f"   Top 5 eigenvalues (real part): {np.sort(eigs_A.real)[::-1][:5]}")
    
    print("\n=== STAGE 2: Learning B from Equilibrium Data ===")
    # Residual: Z_next - A_d @ Z = B_d @ U (at equilibrium, N term small)
    residual_eq = Z_eq_next - Z_eq @ A_discrete.T  # (N_eq, r)
    
    # Regression: residual = B_d @ U
    lhs_B = U_eq.T @ U_eq + alpha
    rhs_B = U_eq.T @ residual_eq
    B_discrete = rhs_B.T / lhs_B  # (r, 1)
    
    # Validate A+B on equilibrium data
    Z_eq_pred = Z_eq @ A_discrete.T + U_eq @ B_discrete.T
    SS_res_AB = np.sum((Z_eq_next - Z_eq_pred)**2)
    SS_tot_AB = np.sum((Z_eq_next - Z_eq_next.mean(0))**2)
    R2_AB = 1 - SS_res_AB / SS_tot_AB
    print(f"   Stage 2 R² (A+B, equilibrium): {R2_AB:.4f}")
    print(f"   |B_discrete|: {np.linalg.norm(B_discrete):.4f}")
    
    # Convert to continuous
    B_cont = B_discrete / dt
    print(f"   |B_cont|: {np.linalg.norm(B_cont):.4f}")
    
    print("\n=== STAGE 3: Learning N from Full Controlled Data ===")
    # Residual: Z_next - A_d @ Z - B_d @ U = N_d @ (Z * U)
    residual_ctrl = Z_ctrl_next - Z_ctrl @ A_discrete.T - U_ctrl @ B_discrete.T
    
    # Bilinear regressor: Z * U
    Z_times_U = Z_ctrl * U_ctrl  # (N_ctrl, r)
    
    # Regression: residual = N_d @ (Z * U)
    lhs_N = Z_times_U.T @ Z_times_U + alpha * np.eye(r)
    rhs_N = Z_times_U.T @ residual_ctrl
    N_discrete = np.linalg.solve(lhs_N, rhs_N).T  # (r, r)
    
    # Validate full model on controlled data
    Z_ctrl_pred = Z_ctrl @ A_discrete.T + U_ctrl @ B_discrete.T + Z_times_U @ N_discrete.T
    SS_res_ABN = np.sum((Z_ctrl_next - Z_ctrl_pred)**2)
    SS_tot_ABN = np.sum((Z_ctrl_next - Z_ctrl_next.mean(0))**2)
    R2_ABN = 1 - SS_res_ABN / SS_tot_ABN
    print(f"   Stage 3 R² (A+B+N, full controlled): {R2_ABN:.4f}")
    print(f"   |N_discrete|: {np.linalg.norm(N_discrete):.4f}")
    
    # Convert to continuous
    N_cont = N_discrete / dt
    
    print("\n=== STAGED LEARNING SUMMARY ===")
    print(f"   R² improvement: A={R2_A:.3f} → A+B={R2_AB:.3f} → A+B+N={R2_ABN:.3f}")
    print(f"   |A_cont|={np.linalg.norm(A_cont):.2f}, |B_cont|={np.linalg.norm(B_cont):.4f}, |N_cont|={np.linalg.norm(N_cont):.4f}")
    
    # Define Target State (for drift check and reduction)
    target_raw = np.array([0., 0., 0., 0.]).reshape(1, -1)
    z_target = get_features(target_raw).flatten()  # 305-dim target
    
    # SPARSIFICATION on continuous matrices (SINDy-style)
    threshold_A = 0.01 * np.max(np.abs(A_cont))
    threshold_B = 0.01 * np.max(np.abs(B_cont)) if np.max(np.abs(B_cont)) > 0 else 0
    threshold_N = 0.01 * np.max(np.abs(N_cont)) if np.max(np.abs(N_cont)) > 0 else 0
    
    A_cont[np.abs(A_cont) < threshold_A] = 0.0
    B_cont[np.abs(B_cont) < threshold_B] = 0.0
    N_cont[np.abs(N_cont) < threshold_N] = 0.0
    
    n_sparse_A = np.sum(A_cont == 0)
    n_sparse_B = np.sum(B_cont == 0)
    n_sparse_N = np.sum(N_cont == 0)
    print(f"Sparsified Continuous: A={n_sparse_A}/{A_cont.size}, B={n_sparse_B}/{B_cont.size}, N={n_sparse_N}/{N_cont.size}")
    
    print(f"Continuous Learned: |Ac|={np.linalg.norm(A_cont):.2f}, |Bc|={np.linalg.norm(B_cont):.2f}, |Nc|={np.linalg.norm(N_cont):.2f}")
    
    # ============================================
    # TENSOR-BASED MODEL REDUCTION (Control-Aware)
    # ============================================
    # ============================================
    # TENSOR-BASED MODEL REDUCTION (Control-Aware)
    # ============================================
    skip_reduction = False  # Use reduced model for speed
    
    if not skip_reduction:
        print("\nApplying Control-Aware Tensor Reduction...")
        # ... (Tensor code) ... (I'll comment it out or put in else block)
        # Construct the Augmented Matrix [A | N | B_scaled]
        scale_B = 100.0 
        M_unfold = np.hstack([A_cont, N_cont, B_cont * scale_B]) # (r, 2r + 1)
        U, S, Vt = np.linalg.svd(M_unfold, full_matrices=False)
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        k_reduced = np.searchsorted(cumulative_energy, 0.999) + 1 
        k_reduced = min(k_reduced, 25)
        k_reduced = max(k_reduced, 6)
        print(f"   Tensor SVD: Reducing {r} -> {k_reduced} dims (99.9% energy)")
        
        V_k = U[:, :k_reduced]  # (r, k)
        A_reduced = V_k.T @ A_cont @ V_k
        B_reduced = V_k.T @ B_cont
        N_reduced = V_k.T @ N_cont @ V_k
        
        print(f"   |A_red|={np.linalg.norm(A_reduced):.2f}, |B_red|={np.linalg.norm(B_reduced):.2f}")
        
        # Update matrices
        z_target_reduced_func = lambda z: V_k.T @ (z - z_target)
        r = k_reduced
        A_cont = A_reduced
        B_cont = B_reduced
        N_cont = N_reduced
    else:
        print("\nSKIPPING REDUCTION: Using Full 305-dim Model")
        V_k = np.eye(r) # Identity projection
        z_target_reduced_func = lambda z: (z - z_target)
        # Matrices stay as A_cont, B_cont, N_cont
    
    # Check Drift at Target
    # Ideally A_cont @ z_target should be 0 (equilibrium)
    # But z_target is just feature mapping of [0,0,0,0].
    # In staged learning, A was learned from autonomous data where u=0.
    # If 0 is equilibrium, drift should be small.
    drift = A_cont @ z_target_reduced_func(z_target) # Should be 0 by definition if z_target is ref
    # Wait, z_target_reduced_func(z_target) is 0. 
    # We should check A_cont @ z_target (absolute drift).
    # But our dynamics are in ERROR coordinates relative to z_target?
    # No, model learned on raw Z.
    # z_dot = A z ...
    # At equilibrium x*, z* = phi(x*). z_dot should be 0.
    drift_val = np.linalg.norm(A_cont @ (z_target if skip_reduction else (V_k.T @ z_target)))
    print(f"DRIFT CHECK: |A @ z_target| = {drift_val:.6f} (Should be small for equilibrium)")

    # 6. Discrete SDRE Setup - PRINCIPLED APPROACH
    print("Setting up Discrete SDRE Control (Principled)...")
    
    # Target (Upright)
    # target_raw = np.array([0., 0., 0., 0.]).reshape(1, -1) # Already defined above? No.
    # We need to redefine it here or use existing

    
    # 6. Discrete SDRE Setup - PRINCIPLED APPROACH
    print("Setting up Discrete SDRE Control (Principled)...")
    
    # Target (Upright)
    target_raw = np.array([0., 0., 0., 0.]).reshape(1, -1)
    z_target = get_features(target_raw).flatten()  # 305-dim target
    
    # SVD projection already handles error coordinates:
    # z_reduced = V_k.T @ (z - z_target)
    # This is inherently in error coordinates since z_target is constant
    
    # The SVD-reduced A_cont, B_cont, N_cont are already set above
    
    # 5b. SPECTRAL STABILIZATION (Crucial for Infinite Dim Kernels)
    # RBF Kernels often induce spurious high-frequency unstable modes.
    # Real CartPole has exactly 1 unstable mode (Gravity).
    # We filter A_cont (or A_reduced) to enforce this physical prior.
    
    print("\nApplying Spectral Stabilization to Control Matrix...")
    # Work on the matrix actually used for control (A_cont which is A_reduced if skipped=False)
    evals, evecs = np.linalg.eig(A_cont)
    
    # Sort by Real part descending
    idx = np.argsort(evals.real)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    print(f"   Original Top 5 Eigs: {evals[:5]}")
    
    # Stabilization Logic:
    # Keep only the #1 unstable mode (Gravity).
    # Damping: Force all other unstable modes to have negative real part.
    # Note: Complex conjugates must be handled symmetrically? 
    # Yes, simply checking Re > 0 and index > 0.
    
    n_unstable_phys = 1
    evals_new = evals.copy()
    
    # Counter for stabilized modes
    n_clamped = 0
    
    for i in range(len(evals)):
        if evals[i].real > 0:
            if i < n_unstable_phys:
                # Keep the dominant instability (Gravity)... BUT CLAMP IT
                # Real CartPole ~ sqrt(9.8) = 3.13. 
                # Learned model often exaggerates this (e.g. 29.0).
                # We clamp the real part to be physically reasonable (e.g. max 4.0).
                
                real_part = min(evals[i].real, 4.0)
                evals_new[i] = real_part + 1j * evals[i].imag
                
            else:
                # Clamp spurious instability
                # Reflect to stable LHS or just set to small decay?
                # Reflection: +x -> -x
                evals_new[i] = -0.1 + 1j * evals[i].imag
                n_clamped += 1
                
    print(f"   Clamped {n_clamped} spurious unstable modes.")
    print(f"   New Top 5 Eigs: {evals_new[:5]}")
    
    # Reconstruct A
    # A = V * diag(w) * V_inv
    A_stabilized = (evecs @ np.diag(evals_new) @ np.linalg.inv(evecs)).real
    
    # Check reconstruction error
    rec_err = np.linalg.norm(A_stabilized - A_cont)
    print(f"   Reconstruction Change Norm: {rec_err:.4f}")
    
    # Replace for Control
    A_cont = A_stabilized
    
    # No need to re-learn in error coordinates

    
    print(f"Using SVD-reduced {r}-dim system for control")
    print(f"Fixed Point Check: projecting target gives {np.linalg.norm(V_k.T @ (z_target - z_target)):.6f}")
    
    # PRINCIPLED COST: Q = I, R = I (no hand-tuning)
    Q_z = np.eye(r)
    R_z = np.eye(1)  
    
    # 7. Multi-Frequency Simulation (Matching Double Well)  
    
    # 7. Multi-Frequency Simulation (Matching Double Well)
    # Feature Centering Applied - No Feedforward Needed
    u_ff = np.zeros((1, 1))
    print(f"Feature Centering Applied. u_ff set to 0.")

    print("Simulating Control Frequency Impact...")
    
    # Scenarios: (Name, Hold Steps, Color)
    # Base dt = 0.02s (Actually env.dt=0.005, so 'hold' scales that)
    # 50Hz -> dt=0.02s -> hold=4? 
    # Wait, earlier I found env.dt=0.005.
    # To get 50Hz (0.02s), we need hold=4.
    # To get 10Hz (0.10s), we need hold=20.
    # Let's update holds to be realistic Hz.
    
    scenarios = [
        # hold=1 means control at every physics step (env.dt=0.005s → 200Hz)
        {"name": "200Hz (hold=1)", "hold": 1, "color": "blue"}
    ]
    
    plt.figure(figsize=(14, 10))
    
    # Prepare Plotting
    ax_x = plt.subplot(3, 1, 1)
    ax_th = plt.subplot(3, 1, 2)
    ax_u = plt.subplot(3, 1, 3)
    
    # Prepare Plotting
    ax_x = plt.subplot(3, 1, 1)
    ax_th = plt.subplot(3, 1, 2)
    ax_u = plt.subplot(3, 1, 3)
    
    final_traj = [] # Default empty list

    
    for scen in scenarios:
        name = scen["name"]
        hold = scen["hold"]
        clr = scen["color"]
        print(f"--- Running {name} (hold={hold}) ---")
        
        env.reset()
        # Start NEAR UPRIGHT (θ ≈ 10°) to test local stabilization
        state = np.array([0.0, 0.0, 0.17, 0.0])  # θ ≈ 10°
        env.state = state
        
        traj = [state]
        actions = []
        
        # Scaling Cost for Frequency? 
        # In discrete LQR, Cost is per step.
        # If we step 5x slower, maybe Q should be 5x larger? 
        # Let's keep Q constant for fair "State Deviation" penalty comparison.
        # But R needs to adjust? No, keep constant to see authority limits.
        
        # But R needs to adjust? No, keep constant to see authority limits.
        
        T_sim_steps = 3000 # 3000 * 0.005 = 15.0s (Extended for verification)
        
        u_val = 0.0
        
        # Use CONTINUOUS dynamics with CARE (no re-discretization needed)
        dt_ctrl = env.dt * hold
        print(f"  Using CARE with dt_ctrl={dt_ctrl:.4f}s (for hold purposes only)...")
        
        for t in range(T_sim_steps):
            x_curr = traj[-1]
            
            # Control Update (Zero Order Hold)
            if t % hold == 0:
                # Get Features in original 305-dim space
                z_curr_full = get_features(x_curr.reshape(1, -1)).flatten()
                
                # Project to reduced SVD space
                z_curr = V_k.T @ (z_curr_full - z_target)  # Reduced error coordinates
                
                # SDRE: State-dependent B in REDUCED space
                B_eff = B_cont + N_cont @ z_curr.reshape(-1, 1)
                
                try:
                    # Solve CONTINUOUS ARE (CARE) in reduced space
                    P_curr = scipy.linalg.solve_continuous_are(A_cont, B_eff, Q_z, R_z)
                    K_gain = np.linalg.solve(R_z, B_eff.T @ P_curr)
                except:
                    K_gain = np.zeros((1, r))
                    
                # LQR Feedback Only (Centered Model)
                u_fb = -K_gain @ z_curr
                u_val = float(u_fb[0])
                
                
                # NaN protection and reasonable limits
                if np.isnan(u_val) or np.isinf(u_val):
                    u_val = 0.0
                u_val = np.clip(u_val, -500, 500)
                
                if t % 500 == 0:
                     print(f"    Step {t}: Th={x_curr[2]*180/np.pi:.1f}, x={x_curr[0]:.2f}, U={u_val:.2f}, |K|={np.linalg.norm(K_gain):.1e}")
            
            # Physics Step (Base Freq)
            env.state = x_curr 
            next_s, _, _, _ = env.step(u_val) # dt=0.02
            
            traj.append(next_s)
            actions.append(u_val)
        
        traj = np.array(traj)
        actions = np.array(actions)
        
        if "200Hz" in name:
            final_traj = traj
        
        # Plot
        time = np.arange(len(traj)) * env.dt
        ax_x.plot(time, traj[:, 0], label=f"{name} x", color=clr)
        ax_th.plot(time, traj[:, 2], label=f"{name} theta", color=clr)
        ax_u.plot(time[:-1], actions, label=f"{name} u", color=clr, alpha=0.5)

    # Finalize Plots
    ax_x.axhline(2.4, color='k', ls=':')
    ax_x.axhline(-2.4, color='k', ls=':')
    ax_x.set_ylabel("Position (x)")
    ax_x.set_ylim([-6, 6])
    ax_x.legend()
    ax_x.set_title("RKHS-KRONIC: Frequency Comparison")
    
    ax_th.axhline(np.pi, color='r', ls='--')
    ax_th.axhline(0, color='g', ls='--') # Target
    ax_th.set_ylabel("Angle (rad)")
    ax_th.legend()
    
    ax_u.set_ylabel("Control (N)")
    ax_u.legend()
    
    plt.tight_layout()
    plt.savefig('rkhs_cartpole_freq_compare.png')
    print("Saved comparison plot.")

    # Animation (High Freq)
    save_animation(final_traj, filename='rkhs_cartpole_swingup.gif', dt=env.dt)

if __name__ == "__main__":
    main()
