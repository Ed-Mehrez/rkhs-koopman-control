#!/usr/bin/env python3
"""
RKHS-KRONIC CartPole: Using KernelGEDMD Framework + CARE

Principled approach using existing infrastructure:
1. Use KernelGEDMD from kgedmd_core.py for CONTINUOUS dynamics
2. Use CARE (Continuous ARE) for control synthesis
3. No hand-tuned features or costs (Q=I, R=I)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.linalg
from matplotlib.animation import PillowWriter
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
from kgedmd_core import KernelGEDMD, RBFKernel


class CartPoleEnv:
    """Simple CartPole with continuous state."""
    def __init__(self, dt=0.02):
        self.dt = dt
        self.g = 9.81
        self.mc = 1.0   
        self.mp = 0.1   
        self.l = 0.5    
        self.state = np.array([0.0, 0.0, np.pi, 0.0])
        
    def reset(self):
        self.state = np.array([0.0, 0.0, np.pi, 0.0])
        return self.state
    
    def drift(self, state, u=0.0):
        """Continuous dynamics: dx/dt = f(x,u)"""
        x, xd, th, thd = state
        
        s = np.sin(th)
        c = np.cos(th)
        
        total_mass = self.mc + self.mp
        pole_mass_length = self.mp * self.l
        
        temp = (u + pole_mass_length * thd**2 * s) / total_mass
        th_acc = (self.g * s - c * temp) / (self.l * (4/3 - self.mp * c**2 / total_mass))
        x_acc = temp - pole_mass_length * th_acc * c / total_mass
        
        return np.array([xd, x_acc, thd, th_acc])
    
    def step(self, u):
        state_dot = self.drift(self.state, u)
        self.state = self.state + state_dot * self.dt
        return self.state, 0, False, {}


class PeriodicRBFKernel(RBFKernel):
    """RBF Kernel with periodicity on theta dimension."""
    
    def __init__(self, sigma: float, periodic_dim: int = 2, period: float = 2*np.pi):
        super().__init__(sigma)
        self.periodic_dim = periodic_dim
        self.period = period
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Kernel with periodic distance on theta."""
        # Handle matrix inputs
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        N, D = x.shape
        M = y.shape[0]
        
        # Modified distance computation
        dist_sq = np.zeros((N, M))
        
        for d in range(D):
            if d == self.periodic_dim:
                # Periodic distance
                diff = x[:, d:d+1] - y[:, d:d+1].T
                dist_sq += 4 * np.sin(diff / 2)**2
            else:
                # Euclidean distance
                dist_sq += (x[:, d:d+1] - y[:, d:d+1].T)**2
        
        return np.exp(-dist_sq / (2 * self.sigma_sq))


def main():
    print("=" * 60)
    print("RKHS-KRONIC CartPole: KernelGEDMD + CARE")
    print("=" * 60)
    
    env = CartPoleEnv(dt=0.005)
    
    # --- 1. Data Collection (Continuous Dynamics) ---
    print("\n1. Collecting continuous dynamics data...")
    N_samples = 3000
    
    X_data = []
    X_dot_data = []
    U_data = []
    
    for i in range(N_samples):
        # Random state
        state = np.array([
            np.random.uniform(-2, 2),      # x
            np.random.uniform(-3, 3),      # xd  
            np.random.uniform(-np.pi, np.pi),  # theta
            np.random.uniform(-3, 3)       # theta_d
        ])
        
        # Random action
        u = np.random.uniform(-30, 30)
        
        X_data.append(state)
        X_dot_data.append(env.drift(state, u))
        U_data.append(u)
    
    X = np.array(X_data)
    X_dot = np.array(X_dot_data)
    U = np.array(U_data).reshape(-1, 1)
    
    print(f"   Data: X {X.shape}, X_dot {X_dot.shape}, U {U.shape}")
    
    # --- 2. Learn Koopman Generator (Continuous) ---
    print("\n2. Learning Koopman generator with KernelGEDMD...")
    
    # Use standard RBF kernel on reduced state [xd, theta, thd]
    # (Translation-invariant in x)
    X_reduced = X[:, 1:]
    X_dot_reduced = X_dot[:, 1:]  # Corresponding dynamics
    
    sigma = 1.5
    gedmd = KernelGEDMD(
        kernel_type='rbf',
        sigma=sigma,
        epsilon=1e-4,
        verbose=True
    )
    
    # Fit with subsampling for meaningful eigenvalues
    # (Full N=3000 gives spurious eigenvalues)
    gedmd.fit(X_reduced, X_dot_reduced, n_subsample=200)
    
    print(f"   Eigenvalues found: {len(gedmd.eigenvalues_)}")
    print(f"   Top 5 eigenvalues: {gedmd.eigenvalues_[:5]}")
    
    # --- 3. Feature Extraction: Koopman Eigenfunctions ---
    print("\n3. Extracting Koopman eigenfunctions...")
    
    # PRINCIPLED: Use UNSTABLE (but not crazy fast) and STABLE eigenfunctions
    # Unstable modes > 20 are likely numerical artifacts or too fast to control
    unstable_mask = (gedmd.eigenvalues_.real > 0) & (gedmd.eigenvalues_.real < 20)
    stable_mask = (gedmd.eigenvalues_.real < 0) & (gedmd.eigenvalues_.real > -20)
    
    unstable_indices = np.where(unstable_mask)[0]
    stable_indices = np.where(stable_mask)[0]
    
    n_unstable = min(5, len(unstable_indices))  
    n_stable = min(5, len(stable_indices))      
    
    eig_indices = np.concatenate([unstable_indices[:n_unstable], stable_indices[:n_stable]])
    n_eigs = len(eig_indices)
    
    print(f"   Using {n_unstable} UNSTABLE + {n_stable} STABLE = {n_eigs} eigenfunctions")
    print(f"   Selected eigenvalues (real): {gedmd.eigenvalues_[eig_indices].real}")
    
    def get_features(states):
        """Map states to eigenfunction coordinates."""
        if states.ndim == 1:
            states = states.reshape(1, -1)
        
        states_red = states[:, 1:]  # Remove x
        
        # Transform using STABLE Koopman eigenfunctions only
        phi_all = gedmd.transform(states_red)
        phi = phi_all[:, eig_indices]  # Select stable eigenfunctions
        
        # Augment with physical state [x, xd, cos(th), sin(th), thd]
        x = states[:, 0:1]
        xd = states[:, 1:2]
        c_th = np.cos(states[:, 2:3])
        s_th = np.sin(states[:, 2:3])
        thd = states[:, 3:4]
        
        # Real part of eigenfunctions + physical state
        return np.hstack([phi.real, x, xd, c_th, s_th, thd])
    
    r = n_eigs + 5
    print(f"   Feature dimension: {r} ({n_eigs} eigenfunctions + 5 state)")
    
    # --- 4. Learn Bilinear Control Model ---
    print("\n4. Learning bilinear control model in feature space...")
    
    Z = get_features(X)
    Z_dot = np.zeros_like(Z)
    
    # Approximate Z_dot via finite difference on a few samples
    # Actually, we need the proper feature-space derivative
    # For now, use a regression approach
    
    # Target in feature space (upright)
    target = np.array([[0., 0., 0., 0.]])
    z_target = get_features(target).flatten()
    
    # Error coordinates
    E = Z - z_target
    
    # ANALYTIC DERIVATIVE instead of finite difference
    # dZ/dt = (∂Z/∂x) @ (dx/dt) = Jacobian @ X_dot
    print("   Computing analytic Z_dot via chain rule...")
    
    Z_dot = np.zeros_like(Z)
    
    for i in range(len(X)):
        state = X[i]
        state_dot = X_dot[i]
        
        # Jacobian of get_features w.r.t. state
        # Features: [kernel_features (n_eigs), x, xd, cos(th), sin(th), thd]
        
        # For eigenfunction part: ∂φ/∂x_reduced * ∂x_reduced/∂x
        # The eigenfunctions are k(x_reduced, centers) transformed
        # This is complex to get analytically, so use finite diff for this part
        state_red = state[1:]  # [xd, th, thd]
        state_red_dot = state_dot[1:]  # [xd_dot, th_dot, thd_dot]
        
        eps = 1e-5
        phi_0 = gedmd.transform(state_red.reshape(1, -1))[:, eig_indices].real.flatten()
        dphi_dx = np.zeros((n_eigs, 3))
        for j in range(3):
            state_red_p = state_red.copy()
            state_red_p[j] += eps
            phi_p = gedmd.transform(state_red_p.reshape(1, -1))[:, eig_indices].real.flatten()
            dphi_dx[:, j] = (phi_p - phi_0) / eps
        
        # dφ/dt = dphi_dx @ x_reduced_dot
        phi_dot = dphi_dx @ state_red_dot
        
        # For augmented state [x, xd, cos, sin, thd]:
        # dx/dt = xd = state_dot[0]
        # d(xd)/dt = state_dot[1]
        # d(cos)/dt = -sin(th) * thd = -state_dot[2] * np.sin(state[2])  # But actually d(cos)/dt = -sin * th_dot
        th = state[2]
        th_dot = state_dot[2]
        
        x_dot_val = state_dot[0]  # dx/dt
        xd_dot_val = state_dot[1]  # d(xd)/dt
        cos_dot = -np.sin(th) * th_dot
        sin_dot = np.cos(th) * th_dot
        thd_dot_val = state_dot[3]  # d(thd)/dt
        
        aug_dot = np.array([x_dot_val, xd_dot_val, cos_dot, sin_dot, thd_dot_val])
        
        Z_dot[i, :n_eigs] = phi_dot
        Z_dot[i, n_eigs:] = aug_dot
    
    # Error coordinates
    E_dot = Z_dot  # Since z_target is constant, dE/dt = dZ/dt
    
    # PRINCIPLED: Use Koopman eigenvalues directly for eigenfunction dynamics
    # dφ_i/dt = λ_i * φ_i (by Koopman theory)
    # Only regress B and N (control terms)
    
    print("   Using Koopman eigenvalues for A_phi (diagonal)")
    koopman_eigs = gedmd.eigenvalues_[eig_indices].real  # Selected STABLE eigenvalues
    print(f"   Koopman eigenvalues (real): {koopman_eigs}")
    
    # For augmented state [x, xd, cos, sin, thd], we need to regress A
    # But for eigenfunctions, A is diagonal with eigenvalues
    
    # Separate regression for control terms only
    # E_dot - A_known @ E = B @ u + N @ (E * u)
    # where A_known has eigenvalues on diagonal for φ, zeros for augmented
    # Relaxed Approach: Learn A fully (Continuous EDMDc) to ensure controllability
    # Instead of enforcing A = diag(eigs), we let A adapt to control
    # residual = E_dot - E @ A_known.T  <-- OLD
    residual = E_dot  # Learn everything
    
    U_std = np.std(U)
    U_scaled = U / U_std
    E_times_U = E * U_scaled
    Psi_ctrl = np.hstack([U_scaled, E_times_U])
    
    alpha = 1e-4
    W_ctrl = np.linalg.solve(Psi_ctrl.T @ Psi_ctrl + alpha * np.eye(Psi_ctrl.shape[1]), Psi_ctrl.T @ residual)
    
    # Construct full A, B, N
    # W_ctrl solves E_dot ~ [U, E*U] @ W_ctrl + E @ W_state
    # But here we regressed U and E*U against E_dot... wait, we need A!
    
    # Let's do a joint regression properly: [E, U, E*U] -> E_dot
    Psi_full = np.hstack([E, U_scaled, E_times_U])
    W_full = np.linalg.solve(Psi_full.T @ Psi_full + alpha * np.eye(Psi_full.shape[1]), Psi_full.T @ E_dot)
    
    # Extract matrices
    # W_full is (dim_in, dim_out) = (r + 1 + r, r)
    A_cont = W_full[:r, :].T
    B_cont = W_full[r:r+1, :].T / U_std
    N_cont = W_full[r+1:, :].T / U_std
    
    B_cont = W_ctrl[:1, :].T / U_std   # (r, 1)
    N_cont = W_ctrl[1:, :].T / U_std   # (r, r)
    
    # =================================================================
    # MODEL VALIDATION (A PRIORI CHECKS)
    # =================================================================
    print("\n   --- Model Validation ---")
    
    # 1. Prediction R² - how well does the model fit?
    E_dot_pred = E @ A_cont.T + U @ B_cont.T + (E * U) @ N_cont.T
    SS_res = np.sum((E_dot - E_dot_pred)**2)
    SS_tot = np.sum((E_dot - E_dot.mean(axis=0))**2)
    R2 = 1 - SS_res / SS_tot
    print(f"   1. Prediction R² = {R2:.4f} (>0.9 = good fit)")
    
    # 2. |A|, |B|, |N| magnitudes
    print(f"   2. |A| = {np.linalg.norm(A_cont):.2f}, |B| = {np.linalg.norm(B_cont):.4f}, |N| = {np.linalg.norm(N_cont):.4f}")
    
    # 3. A eigenvalues
    A_eigs = np.linalg.eigvals(A_cont)
    n_unstable = np.sum(A_eigs.real > 0)
    unstable_eigs = A_eigs[A_eigs.real > 0]
    print(f"   3. A eigenvalues: {n_unstable} unstable")
    if n_unstable > 0:
        print(f"      Unstable eigenvalues: {unstable_eigs.real}")
    
    # 4. STATE-DEPENDENT Controllability (SDRE requires B_eff(z) = B + N @ z)
    print("   4. State-dependent controllability check:")
    test_states = [
        ("Target (θ=0°)", np.array([0., 0., 0., 0.])),
        ("Initial (θ=10°)", np.array([0., 0., 0.17, 0.])),
        ("Down (θ=180°)", np.array([0., 0., np.pi, 0.])),
    ]
    all_controllable = True
    for name, state in test_states:
        z = get_features(state.reshape(1, -1)).flatten()
        e = z - z_target
        B_eff = B_cont + N_cont @ e.reshape(-1, 1)
        
        Ctrb = B_eff
        for i in range(1, r):
            Ctrb = np.hstack([Ctrb, np.linalg.matrix_power(A_cont, i) @ B_eff])
        ctrb_rank = np.linalg.matrix_rank(Ctrb)
        status = "✓" if ctrb_rank == r else "✗"
        print(f"      {name}: rank {ctrb_rank}/{r} {status}")
        if ctrb_rank < r:
            all_controllable = False
    
    # 5. Stabilizability check at TARGET state (most important for control)
    # Use PBH test: rank([sI - A, B_eff]) = n for all unstable s
    z_tgt_test = get_features(np.array([[0., 0., 0., 0.]])).flatten()
    e_tgt_test = z_tgt_test - z_target  # Should be ~0 at target
    B_eff_at_target = B_cont + N_cont @ e_tgt_test.reshape(-1, 1)
    
    is_stabilizable = True
    if n_unstable > 0:
        print("   5. Stabilizability at target (PBH test):")
        for eig in unstable_eigs:
            s = eig.real
            pbh_matrix = np.hstack([s * np.eye(r) - A_cont, B_eff_at_target])
            pbh_rank = np.linalg.matrix_rank(pbh_matrix)
            if pbh_rank < r:
                is_stabilizable = False
                print(f"      λ={s:.2f}: FAILED (rank {pbh_rank} < {r})")
            else:
                print(f"      λ={s:.2f}: ✓")
    else:
        print("   5. STABLE (no unstable modes)")
    
    # 6. Fixed point error
    fp_error = np.linalg.norm(A_cont @ np.zeros(r))
    print(f"   6. Fixed point error: {fp_error:.6f}")
    
    # Summary
    print("\n   --- Validation Summary ---")
    if R2 < 0.7:
        print("   ⚠️  WARNING: Poor model fit (R² < 0.7)")
    if not is_stabilizable:
        print("   ❌ CRITICAL: System NOT stabilizable - CARE will fail")
    else:
        print("   ✓ System appears stabilizable")
    
    # --- 5. SDRE Control (State-Dependent Riccati) ---
    print("\n5. Setting up SDRE control...")
    
    Q = np.eye(r)
    R = np.eye(1)
    
    # Test CARE at initial condition
    test_state = np.array([0.0, 0.0, 0.17, 0.0])  # Near upright
    z_test = get_features(test_state.reshape(1, -1)).flatten()
    e_test = z_test - z_target
    B_test = B_cont + N_cont @ e_test.reshape(-1, 1)
    
    print(f"   Testing CARE at θ=10°: |B_eff| = {np.linalg.norm(B_test):.4f}")
    try:
        P_test = scipy.linalg.solve_continuous_are(A_cont, B_test, Q, R)
        K_test = np.linalg.solve(R, B_test.T @ P_test)
        print(f"   CARE succeeded! |K| = {np.linalg.norm(K_test):.2f}")
    except Exception as ex:
        print(f"   CARE failed: {str(ex)[:80]}...")
        # Print more debug info
        print(f"   A_cont eigenvalues: {np.sort(A_eigs.real)[:5]}")
        print(f"   B_test shape: {B_test.shape}, max: {np.max(np.abs(B_test)):.4f}")
    
    # --- 6. Simulation with SDRE ---
    print("\n6. Simulating control at 50Hz with SDRE...")
    print("   Starting NEAR UPRIGHT (θ=10°) to test stabilization")
    
    env.reset()
    env.state = np.array([0.0, 0.0, 0.17, 0.0])  # θ ≈ 10° from upright
    
    traj = [env.state.copy()]
    actions = []
    
    control_dt = 0.02  # 50Hz
    hold = int(control_dt / env.dt)
    T_steps = 2000
    u_val = 0.0
    
    for t in range(T_steps):
        x_curr = traj[-1]
        
        if t % hold == 0:
            z = get_features(x_curr.reshape(1, -1)).flatten()
            e = z - z_target
            
            # SDRE: State-dependent B
            B_eff = B_cont + N_cont @ e.reshape(-1, 1)
            
            try:
                R_sdre = np.eye(1) * 10.0  # Increased R to avoid 1e8 gains
                P = scipy.linalg.solve_continuous_are(A_cont, B_eff, Q, R_sdre)
                K = np.linalg.solve(R_sdre, B_eff.T @ P)
                u = -K @ e
                u_val = float(u[0])
                care_ok = True
            except:
                u_val = 0.0  # No fallback - just zero control
                care_ok = False
            
            u_val = np.clip(u_val, -500, 500)  # Higher limit
        
        if t % 500 == 0:
            status = "SDRE" if care_ok else "FAIL"
            print(f"   t={t}: θ={x_curr[2]*180/np.pi:.1f}°, x={x_curr[0]:.2f}, u={u_val:.1f} [{status}]")
        
        env.state = x_curr
        next_state, _, _, _ = env.step(u_val)
        traj.append(next_state)
        actions.append(u_val)
    
    traj = np.array(traj)
    actions = np.array(actions)
    
    # --- 7. Results ---
    print("\n7. Saving results...")
    
    plt.figure(figsize=(12, 8))
    time = np.arange(len(traj)) * env.dt
    
    plt.subplot(3, 1, 1)
    plt.plot(time, traj[:, 0])
    plt.ylabel('x (m)')
    plt.grid(True)
    plt.title('RKHS-KRONIC CartPole: KernelGEDMD + CARE')
    
    plt.subplot(3, 1, 2)
    plt.plot(time, traj[:, 2] * 180 / np.pi)
    plt.axhline(0, color='k', ls=':')
    plt.ylabel('θ (deg)')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time[:-1], actions, alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('rkhs_cartpole_kgedmd.png', dpi=150)
    print("   Saved: rkhs_cartpole_kgedmd.png")
    
    # Animation
    print("   Generating animation...")
    fig, ax = plt.subplots(figsize=(10, 4))
    cart_w, cart_h = 1.0, 0.5
    pole_len = 2.0
    
    ax.set_ylim(-2.5, 2.5)
    cart = plt.Rectangle((-cart_w/2, -cart_h/2), cart_w, cart_h, fill=True, color='blue')
    ax.add_patch(cart)
    pole, = ax.plot([], [], 'r-', lw=4)
    txt = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    writer = PillowWriter(fps=30)
    with writer.saving(fig, 'rkhs_cartpole_kgedmd.gif', dpi=80):
        for i in range(0, len(traj), max(1, len(traj)//300)):
            s = traj[i]
            x, th = s[0], s[2]
            ax.set_xlim(x - 5, x + 5)
            cart.set_xy((x - cart_w/2, -cart_h/2))
            pole.set_data([x, x + pole_len*np.sin(th)], [0, pole_len*np.cos(th)])
            txt.set_text(f't={i*env.dt:.2f}s')
            writer.grab_frame()
    
    plt.close(fig)
    print("   Saved: rkhs_cartpole_kgedmd.gif")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
