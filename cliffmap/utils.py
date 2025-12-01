import numpy as np

# Optional imports with fallbacks
try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def wang_divergence(X, Y, k=1, distance='euclidean'):
    """
    Compute Wang divergence between two datasets.
    
    Args:
        X (np.ndarray): First dataset of shape (n, d)
        Y (np.ndarray): Second dataset of shape (m, d) 
        k (int): Number of nearest neighbors
        distance (str): Distance metric ('euclidean', 'manhattan', etc.)
        
    Returns:
        float: Wang divergence value
    """
    if not HAS_SKLEARN:
        # Simple fallback - compute mean pairwise distances
        # This is not the exact Wang divergence but a reasonable approximation
        from scipy.spatial.distance import cdist
        
        # Compute all pairwise distances
        dist_XY = cdist(X, Y, metric=distance)
        dist_XX = cdist(X, X, metric=distance)
        
        # Get k-th nearest neighbor distances
        vi = np.sort(dist_XY, axis=1)[:, min(k-1, dist_XY.shape[1]-1)]
        
        # For self-distances, exclude diagonal (distance 0)
        ri = []
        for i in range(len(X)):
            row_dist = dist_XX[i, :]
            row_dist = row_dist[row_dist > 0]  # Exclude self-distance
            if len(row_dist) >= k:
                ri.append(np.sort(row_dist)[k-1])
            else:
                ri.append(np.sort(row_dist)[-1])  # Use furthest if not enough points
        ri = np.array(ri)
        
        # Simple approximation of Wang divergence
        n, d = X.shape
        m = Y.shape[0]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            term_1 = d / n * np.sum(np.log2(np.maximum(vi, 1e-10) / np.maximum(ri, 1e-10)))
            term_2 = np.log2(m / max(n - 1, 1))
        
        return term_1 + term_2
    
    n, d_x = X.shape
    m, d_y = Y.shape
    
    if d_x != d_y:
        raise ValueError('X and Y have different dimensions')
    
    d = d_x
    
    # Find k nearest neighbors of X in Y
    nbrs_y = NearestNeighbors(n_neighbors=k, metric=distance)
    nbrs_y.fit(Y)
    distances_y, _ = nbrs_y.kneighbors(X)
    
    # Find k+1 nearest neighbors of X in X (excluding self)
    nbrs_x = NearestNeighbors(n_neighbors=k+1, metric=distance)
    nbrs_x.fit(X)
    distances_x, _ = nbrs_x.kneighbors(X)
    
    # Extract the k-th neighbor distances (excluding self for X)
    vi = distances_y[:, -1]
    ri = distances_x[:, -1]
    
    # Compute Wang divergence
    term_1 = d / n * np.sum(np.log2(vi / ri))
    term_2 = np.log2(m / (n - 1))
    
    return term_1 + term_2


def distance_wrapped_v2(x1, x2):
    """
    Compute wrapped distance for circular data.
    
    Args:
        x1 (np.ndarray): First set of angular values
        x2 (np.ndarray): Second set of angular values
        
    Returns:
        float: Wrapped distance
    """
    # Convert to numpy arrays if needed
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    # Compute wrapped angular difference
    diff = np.abs(x1 - x2)
    diff = np.minimum(diff, 2 * np.pi - diff)
    
    return np.sqrt(np.sum(diff**2))


def wrap_to_2pi(angles):
    """
    Wrap angles to [0, 2*pi] range.
    
    Args:
        angles (np.ndarray): Input angles
        
    Returns:
        np.ndarray: Wrapped angles
    """
    return np.mod(angles, 2 * np.pi)


def cart2pol(x, y):
    """
    Convert Cartesian coordinates to polar.
    
    Args:
        x (np.ndarray): X coordinates
        y (np.ndarray): Y coordinates
        
    Returns:
        tuple: (theta, rho) arrays
    """
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    return theta, rho


def pol2cart(theta, rho):
    """
    Convert polar coordinates to Cartesian.
    
    Args:
        theta (np.ndarray): Angles
        rho (np.ndarray): Radii
        
    Returns:
        tuple: (x, y) arrays
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y