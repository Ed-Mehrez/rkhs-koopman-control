#!/usr/bin/env python3
"""
High-Dimensional Scaling Experiment: 20-D Kuramoto Synchronization
Using Sparse Local Koopman SDRE

Demonstrates:
- Control of 20-dimensional nonlinear system
- 40-dimensional feature space
- Sparse approximation scaling (N=20 oscillators)
- Synchronization (Order parameter r -> 1)
"""
import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
from matplotlib import animation
import time

print('='*60)
print('Local Koopman SDRE: 20-D KURAMOTO SYNCHRONIZATION')
print('='*60)

class KuramotoNetwork:
    """
    N coupled oscillators.
    State: [θ_1, ..., θ_N] (phases)
    Control: u_i added to frequency of each oscillator
    Dynamics: θ̇_i = ω_i + u_i + (K/N) * Σ_j sin(θ_j - θ_i)
    """
    def __init__(self, N=20, dt=0.05, K=0.5):
        self.N = N
        self.dt = dt
        self.K = K
        # Intrinsic frequencies (random normal)
        self.omega = np.random.normal(0, 0.5, N) 
        self.state = np.zeros(N)
        
    def reset(self):
        # Start desynchronized (random phases)
        self.state = np.random.uniform(-np.pi, np.pi, self.N)
        return self.state.copy()
        
    def step(self, u):
        u = np.clip(u, -5.0, 5.0)
        theta = self.state
        
        # Vectorized coupling term
        # sin(θ_j - θ_i) matrix
        diff = theta[None, :] - theta[:, None]  # [N, N] matrix of θ_j - θ_i
        coupling = (self.K / self.N) * np.sum(np.sin(diff), axis=1)
        
        dtheta = self.omega + u + coupling
        self.state += dtheta * self.dt
        
        # Keep phases in [-pi, pi]
        self.state = (self.state + np.pi) % (2 * np.pi) - np.pi
        
        # Order parameter r = |(1/N) Σ exp(iθ)|
        z = np.mean(np.exp(1j * self.state))
        r = np.abs(z)
        
        return self.state.copy(), r

def state_to_features(x):
    """
    Lift state to 2N features: [cos(θ_1), sin(θ_1), ..., cos(θ_N), sin(θ_N)]
    This handles the periodicity naturally.
    """
    N = len(x)
    feat = np.zeros(2*N)
    feat[0::2] = np.cos(x)
    feat[1::2] = np.sin(x)
    return feat

class SparseHighDimSDRE:
    """
    Sparse Local Koopman SDRE for High-Dimensional Systems
    Uses k-nearest neighbors to scale to N=20 dimensions.
    """
    def __init__(self, N=20, dt=0.05, kernel_width=2.0, n_nearest=200, min_samples=50):
        self.dt = dt
        self.kernel_width = kernel_width
        self.n_nearest = n_nearest
        self.min_samples = min_samples
        self.X_data, self.U_data, self.Y_data = [], [], []
        self.max_buffer = 5000
        self.n_feat = 2 * N
        self.N = N
            
    def collect(self, x, u, x_next):
        self.X_data.append(state_to_features(x))
        self.U_data.append(np.atleast_1d(u))
        self.Y_data.append(state_to_features(x_next))
        if len(self.X_data) > self.max_buffer:
            self.X_data = self.X_data[-self.max_buffer:]
            self.U_data = self.U_data[-self.max_buffer:]
            self.Y_data = self.Y_data[-self.max_buffer:]
            
    def get_control(self, state):
        if len(self.X_data) < self.min_samples:
            return np.random.uniform(-1, 1, self.N)
            
        z = state_to_features(state)
        X, U, Y = np.array(self.X_data), np.array(self.U_data), np.array(self.Y_data)
        
        # Sparse KNN
        dists = np.linalg.norm(X - z, axis=1)
        n = min(self.n_nearest, len(X))
        idx = np.argsort(dists)[:n]
        X_s, U_s, Y_s = X[idx], U[idx], Y[idx]
        
        weights = np.exp(-dists[idx]**2 / (2 * self.kernel_width**2))
        W = np.diag(weights)
        XU = np.hstack([X_s, U_s])
        
        try:
            dim_in = self.n_feat + self.N
            # Regularized LS for high-dim A, B
            AB = np.linalg.solve(XU.T @ W @ XU + 2.0 * np.eye(dim_in), XU.T @ W @ Y_s)
            
            A_cont = (AB[:self.n_feat].T - np.eye(self.n_feat)) / self.dt
            B_cont = AB[self.n_feat:].T / self.dt
            
            # Target: Sync (all phases same, e.g. 0)
            z_target = np.zeros(self.n_feat)
            z_target[0::2] = 1.0 # cos=1
            
            # Penalize deviation from sync (Tuned for stability)
            Q = 50.0 * np.eye(self.n_feat)
            R = 0.1 * np.eye(self.N)
            
            P = solve_continuous_are(A_cont, B_cont, Q, R)
            K = np.linalg.inv(R) @ B_cont.T @ P
            u = -K @ (z - z_target)
        except Exception as e:
            u = np.random.uniform(-0.5, 0.5, self.N)
            
        return np.clip(u, -5.0, 5.0)

# ============================================================================
# Run Experiment
# ============================================================================
if __name__ == "__main__":
    N = 20
    # Weak coupling K=0.5 makes passive sync impossible - control is required
    env = KuramotoNetwork(N=N, dt=0.05, K=0.5) 
    ctrl = SparseHighDimSDRE(N=N, dt=0.05, kernel_width=2.5, n_nearest=300)

    env.reset()
    obs = env.state.copy()

    r_history = []
    phases_history = []
    t_start = time.time()

    print(f'Starting 20-D Kuramoto Control (weak coupling K=0.5)...')
    print(f'Goal: Hold r > 0.9 for extended period (Long Run).')

    max_steps = 10000
    synced_steps = 0

    for t in range(max_steps):
        # 1. Get Control
        u = ctrl.get_control(obs)
        
        # 2. Step Environment
        next_obs, r = env.step(u)
        
        # 3. Learn Online (Store Tuple)
        ctrl.collect(obs.copy(), u, next_obs.copy())
        
        obs = next_obs.copy()
        
        r_history.append(r)
        phases_history.append(obs.copy())
        
        if r > 0.90:
            synced_steps += 1
        
        if t % 200 == 0:
            elapsed = time.time() - t_start
            stability = 100 * synced_steps / (t + 1)
            print(f'  Step {t}: r={r:.3f}, Stability={stability:.1f}%, Samples={len(ctrl.X_data)}')

    total_time = time.time() - t_start
    final_r = r_history[-1]
    stability = 100 * synced_steps / max_steps
    
    print(f'\\nFinal r={final_r:.3f}. Total Time: {total_time:.1f}s')
    print(f'Stability (r > 0.9): {stability:.1f}% of time')


    # ============================================================================
    # Polarization Plot GIF
    # ============================================================================
    print('\\nGenerating Kuramoto GIF...')
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    phases_hist = np.array(phases_history)
    color_cycle = plt.cm.hsv(np.linspace(0, 1, N))

    def render(i):
        ax.clear()
        phases = phases_hist[i]
        # Plot oscillators on unit circle
        ax.scatter(phases, np.ones(N), c=color_cycle, s=100, alpha=0.8)
        
        # Plot order parameter vector (magnitude r, angle psi)
        z = np.mean(np.exp(1j * phases))
        r, psi = np.abs(z), np.angle(z)
        ax.plot([0, psi], [0, r], 'k-', linewidth=3)
        ax.plot(psi, r, 'ko', markersize=8)
        
        ax.set_ylim(0, 1.1)
        ax.set_yticks([])
        ax.set_title(f'Step {i}, r={r_history[i]:.2f}')
        return []

    step = max(1, len(phases_hist) // 200)
    frames = list(range(0, len(phases_hist), step))
    anim = animation.FuncAnimation(fig, render, frames=len(frames), interval=50)
    anim.save('kuramoto_sdre.gif', writer='pillow', fps=20)
    print('Saved kuramoto_sdre.gif')
