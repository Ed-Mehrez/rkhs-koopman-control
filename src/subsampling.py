
import numpy as np
from typing import Tuple, Optional

def subsample(X: np.ndarray, Y: np.ndarray, Z: Optional[np.ndarray], n_subsample: int, method: str, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Subsample the data using various strategies."""
    if verbose:
        print(f"   Subsampling data to {n_subsample} points using '{method}' method...")

    if method == 'random':
        indices = np.random.choice(X.shape[1], n_subsample, replace=False)
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_subsample, random_state=42, n_init=10).fit(X.T)
        indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(X.T - center, axis=1)
            indices.append(np.argmin(distances))
        indices = np.array(indices)
    elif method == 'farthest_point':
        # Optimized O(MN) implementation
        n_samples = X.shape[1]
        indices = [np.random.randint(n_samples)] # Start with random point
        
        # Initialize distances to infinity
        min_dists = np.full(n_samples, np.inf)
        
        for _ in range(n_subsample - 1):
            last_idx = indices[-1]
            # Update distances: min_dist(x, S_new) = min(min_dist(x, S_old), dist(x, last_added))
            # New distances from the last added point to all other points
            new_dists = np.linalg.norm(X - X[:, last_idx:last_idx+1], axis=0)
            min_dists = np.minimum(min_dists, new_dists)
            
            # Select point with maximum minimum distance
            best_idx = np.argmax(min_dists)
            indices.append(best_idx)
            
        indices = np.array(indices)
        
    elif method == 'velocity_weighted_farthest_point':
        # Optimized O(MN) implementation with weights
        n_samples = X.shape[1]
        velocities = np.linalg.norm(Y, axis=0)
        weights = 1 + velocities / (np.max(velocities) + 1e-12)
        
        indices = [np.random.randint(n_samples)]
        min_dists = np.full(n_samples, np.inf)
        
        for _ in range(n_subsample - 1):
            last_idx = indices[-1]
            new_dists = np.linalg.norm(X - X[:, last_idx:last_idx+1], axis=0)
            min_dists = np.minimum(min_dists, new_dists)
            
            # Weighted score
            scores = min_dists * weights
            
            # Don't pick already selected points (dist 0)
            scores[indices] = -1
            
            best_idx = np.argmax(scores)
            indices.append(best_idx)
            
        indices = np.array(indices)
    else:
        raise ValueError(f"Unknown subsampling method: {method}")

    X_sub = X[:, indices]
    Y_sub = Y[:, indices]
    Z_sub = Z[:, :, indices] if Z is not None else None
    
    if verbose:
        print(f"   Subsampled data: X={X_sub.shape}, Y={Y_sub.shape}")

    return X_sub, Y_sub, Z_sub
