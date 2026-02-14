
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image
import io

# Add project root to path
if __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.cartpole_env import CartPoleEnv
from src.kgedmd_core import RBFKernel
from src.kronic_controller import KRONICController

def render_cartpole(state, mode_text=""):
    """
    Renders CartPole state to an RGB array.
    """
    x, x_dot, theta, theta_dot = state
    
    # Dimensions
    cart_width = 1.0
    cart_height = 0.6
    pole_length = 3.0 
    
    fig, ax = plt.subplots(figsize=(6, 4))
    cam_x = x
    ax.set_xlim(cam_x - 4.0, cam_x + 4.0)
    ax.set_ylim(-1.5, 3.5)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Floor
    ax.plot([cam_x - 10.0, cam_x + 10.0], [0, 0], 'k-', lw=1)
    
    # Cart
    cart = plt.Rectangle((x - cart_width/2, 0), cart_width, cart_height, color='black')
    ax.add_patch(cart)
    
    # Pole
    pole_x = [x, x + pole_length * np.sin(theta)]
    pole_y = [cart_height/2, cart_height/2 + pole_length * np.cos(theta)]
    ax.plot(pole_x, pole_y, 'r-', lw=4)
    
    # Title
    ax.set_title(f"RKHS-KRONIC Control | {mode_text}\nx={x:.2f}, theta={np.rad2deg(theta):.1f}")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    img = Image.open(buf)
    return img

def collect_data(n_samples=2000):
    print(f"📊 Collecting {n_samples} training samples...")
    env = CartPoleEnv(dt=0.02)
    env.reset()
    
    X = []
    X_dot = []
    U = []
    
    # Strategy: Visit relevant state space (Swing up + Stabilization)
    # We mix random actions with energy-pumping heuristics to cover the space
    
    for i in range(n_samples):
        # Reset occasionally to cover different start states
        if i % 100 == 0:
            env.reset()
            # 50% Chance: Start near UPGIGHT (Critical for LQR Linearization)
            if np.random.rand() < 0.5:
                # Small perturbation from upright [0, 0, 0, 0]
                env.state = np.random.normal(scale=[0.5, 0.5, 0.2, 0.5])
            else:
                # Random Global State
                env.state = np.random.uniform(low=[-2, -5, -np.pi, -5], high=[2, 5, np.pi, 5])
            
        obs = np.array(env.state)
        
        # Action Strategy: Random + Stabilizing (LQR-like)
        if np.abs(obs[2]) < 0.5:
            # Near upright: Use heuristic stabilizer to generate "good" data
            # PD control: u = Kp*theta + Kd*theta_dot + Kx*x
            u = 20.0 * obs[2] + 5.0 * obs[3] + 2.0 * obs[0] + 1.0 * obs[1]
            # Add noise to explore
            u += np.random.normal(0, 5.0)
        else:
             # Swing up/Random
            if np.random.rand() < 0.3:
                u = np.random.uniform(-30, 30) # Random
            else:
                # Simple energy pumping
                u = 30.0 * np.sign(obs[3] * np.cos(obs[2])) 
                
        u = np.clip(u, -30, 30)
        u_vec = np.array([u])
        
        # Step
        next_obs, _, _, _ = env.step(u)
        
        # Finite difference approx of derivative
        x_dot = (next_obs - obs) / 0.02
        
        X.append(obs)
        X_dot.append(x_dot)
        U.append(u_vec)
        
        env.state = next_obs # Update internal state manually if needed, but step does it.
        
    return np.array(X), np.array(X_dot), np.array(U)

def run_stabilization_test(controller):
    print("\n⚖️ Testing Stabilization (Start Upright)...")
    env = CartPoleEnv(dt=0.02)
    env.reset()
    # Start unstable upright
    env.state = np.array([0.0, 0.0, 0.1, 0.0]) # Slight perturbation
    
    frames = []
    obs = np.array(env.state)
    traj_theta = []
    
    for t in range(500):
        # KRONIC Control
        u = controller.control(obs)
        if isinstance(u, np.ndarray): u = u.item()
        
        u = np.clip(u, -30, 30) # Safety clip
        
        next_obs, _, _, _ = env.step(u)
        obs = next_obs
        traj_theta.append(obs[2])
        
        if t % 5 == 0:
            frames.append(render_cartpole(obs, f"Stabilization (t={t})"))
            
    max_theta = np.max(np.abs(traj_theta))
    print(f"   Max theta deviation: {max_theta:.4f} rad")
    
    if max_theta < 0.5:
        print("✅ Stabilization SUCCESS!")
        frames[0].save('kronic_stabilization.gif', save_all=True, append_images=frames[1:], duration=40, loop=0)
    else:
        print("❌ Stabilization FAILED (Diverged)")

def run_experiment():
    # 1. Collect Data
    X_train, X_dot_train, U_train = collect_data(n_samples=3000)
    
    # 2. Fit KRONIC Controller
    print("\n🚀 Fitting KRONIC Controller...")
    # RBF Kernel bandwidth: crucial tuning
    # Sigma=4.0 seems good for global coverage based on Sig-KKF experience
    kernel = RBFKernel(sigma=4.0) 
    
    target_state = np.array([0, 0, 0, 0]) # Upright
    # High penalty on Theta (state 2) and Theta_dot (state 3)
    cost_weights = {'Q': np.diag([5.0, 0.1, 10.0, 1.0]), 'R': 0.1}
    
    controller = KRONICController(
        kernel=kernel, 
        target_state=target_state, 
        cost_weights=cost_weights,
        verbose=True
    )
    
    # Fit with enforce_stability=FALSE because CartPole is naturally unstable
    controller.fit(X_train, X_dot_train, U_train, dt=0.02, enforce_stability=False)
    
    # 3. Test
    run_stabilization_test(controller)

if __name__ == "__main__":
    run_experiment()
