import numpy as np
import scipy.linalg
import sys
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# --- 1. Environment Definition ---
class ControlledDoubleWellEnv:
    def __init__(self, dt=0.05, noise_strength=0.0):
        self._dt = dt # Increased for proper ID
        self._state_dim = 2
        self._action_dim = 2
        self.noise_strength = noise_strength
        self.state = self.reset()

    @property
    def dt(self): return self._dt
        
    def reset(self, state=None):
        if state is None:
            self.state = np.array([-1.0, 0.0]) + np.random.normal(0, 0.05, 2)
        else:
            self.state = np.array(state)
        return self.state
        
    def gradient(self, x, y):
        # dV/dx = 4x^3 - 4x
        # dV/dy = 2y
        dv_dx = 4 * x**3 - 4 * x
        dv_dy = 2 * y
        return np.array([dv_dx, dv_dy])
        
    def step(self, action):
        u = np.clip(action, -5.0, 5.0)
        x, y = self.state
        grad = self.gradient(x, y)
        
        drift_x = (-grad[0] + u[0]) * self.dt
        drift_y = (-grad[1] + u[1]) * self.dt
        
        noise = self.noise_strength * np.sqrt(self.dt) * np.random.normal(0, 1, 2)
        self.state += np.array([drift_x, drift_y]) + noise
        
        return self.state.copy(), 0, False, {}

# --- 2. Feature Map ---
class PolyFeatureMap:
    def __init__(self, degree=3):
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.dim = None
    def fit(self, X):
        self.poly.fit(X)
        self.dim = self.poly.n_output_features_
        print(f"Polynomial Features (Degree {self.poly.degree}) Dim: {self.dim}")
    def transform(self, x):
        if x.ndim == 1: x = x.reshape(1, -1)
        return self.poly.transform(x).flatten() if x.shape[0]==1 else self.poly.transform(x)

# --- 3. Koopman Dynamics Learner ---
class KoopmanLearner:
    def __init__(self, feature_map, state_dim=2, action_dim=2):
        self.phi = feature_map
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def fit(self, X, U, Xn):
        # Generator EDMD
        dt = 0.05
        Z = self.phi.transform(X)
        Zn = self.phi.transform(Xn)
        self.z_dim = Z.shape[1]
        
        # Generator Target: (Zn - Z)/dt
        gen_target = (Zn - Z) / dt
        
        print(f"Gen Target Stats. Mean: {np.mean(gen_target):.4f}, Std: {np.std(gen_target):.4f}")
        
        # G = [Z, U]
        G = np.hstack([Z, U])
        
        # Ridge (Low Regularization for Low Noise)
        lam = 1.0
        GT_G = G.T @ G
        GT_Y = G.T @ gen_target
        
        print(f"Condition Number G: {np.linalg.cond(G):.2e}")
        
        K_gen_T = scipy.linalg.solve(GT_G + lam * np.eye(G.shape[1]), GT_Y)
        K_gen = K_gen_T.T 
        
        self.Ac = K_gen[:, :self.z_dim]
        self.Bc = K_gen[:, self.z_dim:]
        
        print(f"Learned Ac Norm: {np.linalg.norm(self.Ac):.4f}")
        
        # Discretize (Taylor 1st)
        self.A = np.eye(self.z_dim) + self.Ac * dt
        self.B = (np.eye(self.z_dim) + 0.5 * self.Ac * dt) @ self.Bc * dt # approx
        
    def predict(self, z, u):
        return self.A @ z + self.B @ u

# --- Main ---
if __name__ == "__main__":
    np.random.seed(42)
    # Low Noise Learning
    env = ControlledDoubleWellEnv(dt=0.05, noise_strength=0.01)
    feats = PolyFeatureMap(degree=3)
    
    # 1. Collect Data
    print("Collecting Data...")
    X, U, Xn = [], [], []
    
    # Global Sampling
    for _ in range(5000):
        state_init = np.random.uniform(-2, 2, size=(2,))
        env.state = state_init.copy()
        
        u = np.random.uniform(-3, 3, size=(2,))
        
        # Pre-step append (Correct)
        X.append(state_init.copy())
        U.append(u.copy())
        
        next_state, _, _, _ = env.step(u)
        Xn.append(next_state.copy())
        
    X = np.array(X); U = np.array(U); Xn = np.array(Xn)
    feats.fit(X)
    
    # 2. Learn
    learner = KoopmanLearner(feats)
    learner.fit(X, U, Xn)
    
    # 3. Validate
    val_x = np.linspace(-1.5, 1.5, 50)
    val_X = np.column_stack([val_x, np.zeros_like(val_x)])
    val_U = np.zeros((50, 2))
    
    true_dx = (val_x - val_x**3) * env.dt
    val_Z = feats.transform(val_X)
    val_Zn = learner.predict(val_Z.T, val_U.T).T
    
    # Extract x prediction (assume index 0 is x)
    names = feats.poly.get_feature_names_out(['x', 'y'])
    x_idx = np.where(names == 'x')[0][0]
    
    pred_xn = val_Zn[:, x_idx]
    pred_dx = pred_xn - val_x
    
    mse = np.mean((pred_dx - true_dx)**2)
    null_mse = np.mean((0 - true_dx)**2)
    
    print(f"Validation MSE: {mse:.8f}")
    print(f"Null Model MSE: {null_mse:.8f}")
    
    if mse < null_mse:
        print("SUCCESS: Learned Non-Trivial Dynamics.")
        ratio = null_mse / mse
        print(f"Improvement: {ratio:.1f}x")
    else:
        print("FAILURE: Model is Identity or Worse.")
        
    # Plot
    plt.figure()
    plt.plot(val_x, true_dx/env.dt, 'k--', label='True dx/dt')
    plt.plot(val_x, pred_dx/env.dt, 'r-', label='Pred dx/dt')
    plt.legend()
    plt.savefig('double_well_debug.png')
