import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky
from .utils import wrap_to_2pi, cart2pol, pol2cart
import warnings

# Optional imports with fallbacks
try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class Batch:
    """
    Python implementation of the MATLAB Batch class for CLiFF-map.
    
    This class handles flow field processing for a single location using
    circular-linear statistics and expectation maximization.
    """
    
    # Constants
    THETA_LO = 0
    THETA_HI = 2 * np.pi
    RHO_LO = 0
    RHO_HI = 10
    
    def __init__(self):
        # Input properties
        self.id = None
        self.pose = None
        self.data = None
        self.track_id = None
        self.wind = None
        
        # Mean Shift parameters
        self.bandwidth = None
        self.kernel_type = None
        self.distance_type = None
        self.stop_fraction = None
        self.clusters_ids = None
        self.clusters_means = None
        self.clusters_means_mle = None
        self.cluster_covariance = None
        
        # Output properties
        self.mean = None
        self.cov = None
        self.p = None  # mixing factors
        self.observed = None
        self.p_weight = None  # trust in motion pattern
        self.scale_p = None  # scaled trust weight
        self.q = None  # trust in location
        self.t = None  # trust in reconstructed distribution
        self.r = None  # mean resultant length
        self.bic = None  # Bayesian Information Criterion
        self.aic = None  # Akaike Information Criterion
        
        # Clustering
        self.map_cluster_direction = None
        
        # Diagnostics
        self.rem_nan = 0
        self.rem_smal = 0
        self.rem_red = 0
        self.ridgeline = False
    
    def set_parameters(self, batch_id, pose, data, wind, ridgeline=False):
        """Set batch parameters."""
        self.id = batch_id
        self.pose = pose
        self.wind = wind if wind is not None else 0  # Default wind to 0 if None
        self.ridgeline = ridgeline
        
        if data is not None and len(data) > 0:
            self.add_data(data)
        
        return self
    
    def add_data(self, data):
        """Add velocity data to the batch, removing static objects."""
        # Remove static objects (speed = 0)
        data = data[data[:, 1] != 0]
        # Wrap angles to [0, 2π]
        data[:, 0] = wrap_to_2pi(data[:, 0])
        
        if len(data) > 2:
            self.data = data
        
        return self
    
    def set_parameters_mean_shift(self, stop_fraction, bandwidth):
        """Set Mean Shift algorithm parameters."""
        self.stop_fraction = stop_fraction
        
        if bandwidth == -1:
            # Automatic bandwidth selection
            mean, cov = self.mle(self.data)
            sigma_c = cov[0, 0]
            sigma_l = cov[1, 1]
            
            n = len(self.data)
            h_l = ((4 * sigma_l**5) / (3 * n))**(1/5)
            h_c = ((4 * sigma_c**5) / (3 * n))**(1/5)
            
            if h_l < np.finfo(float).eps or h_c < np.finfo(float).eps:
                h_l += 1e-10
                h_c += 1e-10
            
            self.bandwidth = np.array([[h_c, 0], [0, h_l]])
        else:
            self.bandwidth = bandwidth
        
        return self
    
    def probability(self, velocity):
        """Compute probability of a velocity under the learned model."""
        prob = 0
        if self.mean is None or self.cov is None or self.p is None:
            return prob
        
        n_components = len(self.mean)
        for i in range(n_components):
            for k in range(-self.wind, self.wind + 1):
                wrapped_mean = self.mean[i] + np.array([2 * np.pi * k, 0])
                prob += self.normal(velocity, self.cov[:, :, i], wrapped_mean) * self.p[i]
        
        return prob
    
    def mean_shift_2d(self):
        """Perform 2D Mean Shift clustering on circular-linear data."""
        if self.data is None or len(self.data) == 0:
            return self, None
        
        local_data = self.data.copy()
        data_count = len(self.data)
        shift_data = np.zeros_like(local_data)
        
        # Vectorized mean shift computation
        for i in range(data_count):
            moved = True
            current_point = local_data[i].copy()
            iterations = 0
            max_iterations = 100  # Prevent infinite loops
            
            while moved and iterations < max_iterations:
                old_point = current_point.copy()
                
                # Vectorized distance computation
                data_diff = self.data - old_point  # Shape: (n_data, 2)
                
                # Handle circular wrapping for angle differences
                angle_diff = data_diff[:, 0]
                angle_diff = np.mod(angle_diff + np.pi, 2*np.pi) - np.pi  # Wrap to [-π, π]
                data_diff[:, 0] = angle_diff
                
                # Compute weights for all data points at once
                total_weights = np.zeros(data_count)
                
                # Vectorized computation across winding numbers
                for k in range(-self.wind, self.wind + 1):
                    wrapped_diff = data_diff.copy()
                    wrapped_diff[:, 0] += 2 * np.pi * k
                    
                    # Vectorized multivariate normal PDF
                    try:
                        weights = multivariate_normal.pdf(wrapped_diff, 
                                                        mean=np.zeros(2), 
                                                        cov=self.bandwidth, allow_singular=True)
                        total_weights += weights
                    except (np.linalg.LinAlgError, ValueError):
                        # Fallback for numerical issues
                        regularized_bandwidth = self.bandwidth + np.eye(2) * 1e-6
                        weights = multivariate_normal.pdf(wrapped_diff,
                                                        mean=np.zeros(2),
                                                        cov=regularized_bandwidth, allow_singular=True)
                        total_weights += weights
                
                # Weighted mean calculation
                new_point = self.weighted_mean_cs(self.data, total_weights)
                
                # Check convergence
                distance_moved = self.distance_disjoint(old_point, new_point)
                if (distance_moved[0] < self.bandwidth[0, 0] * self.stop_fraction and
                    distance_moved[1] < self.bandwidth[1, 1] * self.stop_fraction):
                    moved = False
                    shift_data[i] = new_point
                else:
                    current_point = new_point
                    iterations += 1
        
        # Cluster the shifted points
        cluster_centers = []
        cluster_ids = -np.ones(data_count, dtype=int)
        
        for i in range(data_count):
            found = False
            for j, center in enumerate(cluster_centers):
                distance = self.distance_disjoint(center, shift_data[i])
                threshold = self.bandwidth * self.stop_fraction * 10
                
                if (distance[0] < threshold[0, 0] and distance[1] < threshold[1, 1]):
                    found = True
                    cluster_ids[i] = j
                    # Update cluster center
                    cluster_data = shift_data[cluster_ids == j]
                    cluster_centers[j] = self.centroid(cluster_data)
                    break
            
            if not found:
                cluster_centers.append(shift_data[i])
                cluster_ids[i] = len(cluster_centers) - 1
        
        # Prune small clusters and compute statistics
        cluster_centers, cluster_ids = self._prune_small_clusters(
            cluster_centers, cluster_ids)
        
        self.clusters_ids = cluster_ids
        self.clusters_means = np.array(cluster_centers) if cluster_centers else None
        
        # Compute covariances
        if cluster_centers:
            self._compute_cluster_covariances(cluster_ids)
        
        return self, shift_data
    
    def em_algorithm(self):
        """Perform Expectation Maximization to fit Gaussian mixture model."""
        if self.clusters_means is None or len(self.clusters_means) == 0:
            return self
        
        cluster_means = self.clusters_means.copy()
        cluster_covariance = self.cluster_covariance.copy()
        
        self.rem_nan = 0
        self.rem_smal = 0
        
        n_clusters = len(cluster_means)
        p = np.ones(n_clusters) / n_clusters  # Equal initial weights
        l_range = np.arange(-self.wind, self.wind + 1)
        n_data, _ = self.data.shape
        
        # Initialize covariances
        c = np.zeros_like(cluster_covariance)
        for j in range(n_clusters):
            f = np.diag(cluster_covariance[:, :, j])
            # Handle zero or negative values
            f = np.maximum(f, 1e-10)  # Ensure positive values
            f = np.floor(np.log10(f))
            c[:, :, j] = np.diag([10**(f[0]-1), 10**(f[1]-1)])
        
        m = cluster_means.copy()
        old_log_likelihood = 0
        delta = np.inf
        epsilon = 1e-5
        max_iteration = 500
        iteration = 0
        
        while abs(delta) > epsilon and iteration < max_iteration:
            iteration += 1
            
            # E-step: Compute responsibilities
            r = np.zeros((n_data, n_clusters, len(l_range)))
            
            for j in range(n_clusters):
                for l_idx, l in enumerate(l_range):
                    wrapped_mean = m[j] + np.array([2 * np.pi * l, 0])
                    try:
                        r[:, j, l_idx] = p[j] * multivariate_normal.pdf(
                            self.data, mean=wrapped_mean, cov=c[:, :, j], allow_singular=True)
                    except np.linalg.LinAlgError:
                        # Handle singular covariance by using allow_singular=True
                        r[:, j, l_idx] = p[j] * multivariate_normal.pdf(
                            self.data, mean=wrapped_mean, cov=c[:, :, j] + np.eye(2)*1e-6, allow_singular=True)
            
            # Normalize responsibilities
            r[r < np.finfo(float).eps] = 0
            r_sum = np.sum(r, axis=(1, 2), keepdims=True)
            r_sum[r_sum == 0] = 1  # Avoid division by zero
            r = r / r_sum
            r[np.isnan(r)] = 0
            
            # M-step: Update parameters
            for j in range(n_clusters):
                # Update means
                numerator = np.zeros(2)
                for l_idx, l in enumerate(l_range):
                    wrapped_data = self.data - np.array([2 * np.pi * l, 0])
                    numerator += np.sum(wrapped_data * r[:, j, l_idx:l_idx+1], axis=0)
                
                denominator = np.sum(r[:, j, :])
                if denominator > 0:
                    m[j] = numerator / denominator
                
                # Update covariances
                cov_sum = np.zeros((2, 2))
                for l_idx, l in enumerate(l_range):
                    diff = self.data - m[j] - np.array([2 * np.pi * l, 0])
                    weighted_diff = diff * np.sqrt(r[:, j, l_idx:l_idx+1])
                    cov_sum += weighted_diff.T @ weighted_diff
                
                if denominator > 0:
                    c[:, :, j] = cov_sum / denominator
                
                # Update mixing coefficients
                p[j] = np.sum(r[:, j, :]) / n_data
            
            # Regularize covariances to avoid singularity
            for j in range(n_clusters):
                # Add regularization to ensure positive definite matrices
                eigenvals = np.linalg.eigvals(c[:, :, j])
                if np.any(eigenvals <= 0):
                    c[:, :, j] += np.eye(2) * 1e-6
                
                # Additional check for numerical stability
                try:
                    np.linalg.cholesky(c[:, :, j])
                except np.linalg.LinAlgError:
                    c[:, :, j] = np.eye(2) * 0.01  # Reset to identity with small variance
            
            # Compute log-likelihood for convergence check (vectorized)
            log_likelihood = 0
            
            # Vectorized likelihood computation
            data_expanded = self.data[:, np.newaxis, :]  # Shape: (n_data, 1, 2)
            
            for j in range(n_clusters):
                component_likelihood = np.zeros(n_data)
                
                # Compute wrapped means for all l values at once
                for l in l_range:
                    wrapped_mean = m[j] + np.array([2 * np.pi * l, 0])
                    
                    try:
                        # Vectorized PDF computation
                        pdf_vals = multivariate_normal.pdf(
                            self.data, mean=wrapped_mean, cov=c[:, :, j], allow_singular=True)
                        component_likelihood += pdf_vals
                    except (np.linalg.LinAlgError, ValueError):
                        # Fallback with regularized covariance
                        pdf_vals = multivariate_normal.pdf(
                            self.data, mean=wrapped_mean, 
                            cov=c[:, :, j] + np.eye(2)*1e-6, allow_singular=True)
                        component_likelihood += pdf_vals
                
                # Weight by mixing probability
                component_likelihood *= p[j]
                
                if j == 0:
                    total_likelihood = component_likelihood
                else:
                    total_likelihood += component_likelihood
            
            # Compute log-likelihood with numerical stability
            total_likelihood = np.maximum(total_likelihood, 1e-300)  # Avoid log(0)
            log_likelihood = np.sum(np.log(total_likelihood))
            
            delta = log_likelihood - old_log_likelihood
            old_log_likelihood = log_likelihood
        
        # Store results
        self.mean = m
        self.cov = c
        self.p = p
        
        # Compute information criteria
        self._compute_information_criteria()
        
        return self
    
    def sample(self, n):
        """Sample n points from the learned distribution."""
        if self.mean is None or self.cov is None or self.p is None:
            return np.array([])
        
        return self.mvgmm_random(self.mean, self.cov, self.p, n)
    
    def sample_raw(self, n):
        """Sample n points using raw sampling method."""
        return self.sample(n)
    
    def score_fit(self):
        """Compute fit scoring metrics."""
        if self.data is None or self.mean is None:
            return self
        
        # Compute resultant length for each component
        if self.mean is not None:
            self.r = np.zeros(len(self.mean))
            for i in range(len(self.mean)):
                # Convert to complex representation for circular mean
                angles = self.data[:, 0]
                z = np.exp(1j * angles)
                self.r[i] = np.abs(np.mean(z))
        
        return self
    
    # Static methods for distance calculations and statistics
    @staticmethod
    def distance_wrapped_v2(p1, p2):
        """Compute wrapped distance between points in circular-linear space."""
        if p2.ndim == 1:
            p2 = p2.reshape(1, -1)
        
        ad = np.abs(np.angle(np.exp(1j * (p1[0] - p2[:, 0]))))
        ld = np.abs(p1[1] - p2[:, 1])
        return np.sqrt(ad**2 + ld**2)
    
    @staticmethod
    def distance_wrapped(p1, p2):
        """Compute wrapped distance between two points."""
        ad = np.abs(np.angle(np.exp(1j * (p1[0] - p2[0]))))
        ld = np.abs(p1[1] - p2[1])
        return np.sqrt(ad**2 + ld**2)
    
    @staticmethod
    def distance_disjoint(p1, p2):
        """Compute disjoint distance (separate circular and linear components)."""
        ad = np.abs(np.angle(np.exp(1j * (p1[0] - p2[0]))))
        ld = np.abs(p1[1] - p2[1])
        return np.array([ad, ld])
    
    @staticmethod
    def mle(x):
        """Maximum likelihood estimation for circular-linear data."""
        if x.ndim != 2:
            raise ValueError('The arguments should be 2 dimensional')
        
        # Circular component
        c = np.mean(np.cos(x[:, 0]))
        s = np.mean(np.sin(x[:, 0]))
        r = np.sqrt(c**2 + s**2)
        
        if c >= 0:
            cr_m = np.arctan(s / c) if c != 0 else np.pi/2 * np.sign(s)
        else:
            cr_m = np.arctan(s / c) + np.pi
        
        # Linear component
        l_m = np.mean(x[:, 1])
        
        mean = np.array([wrap_to_2pi(cr_m), l_m])
        
        # Covariance estimation
        std = np.sqrt(-2 * np.log(r)) if r > 0 else 1.0
        v_c = std**2
        v_l = np.var(x[:, 1], ddof=1) if len(x) > 1 else 1.0
        
        # Cross-covariance
        c_cross = 0
        for j in range(len(x)):
            angle_diff = np.angle(np.exp(1j * (x[j, 0] - cr_m)))
            c_cross += (x[j, 1] - l_m) * angle_diff
        c_cross = c_cross / (len(x) - 1) if len(x) > 1 else 0
        
        cov = np.array([[v_c, c_cross], [c_cross, v_l]])
        
        return mean, cov
    
    @staticmethod
    def centroid(x):
        """Compute centroid of circular-linear data."""
        if x.ndim != 2:
            raise ValueError('The arguments should be 2 dimensional')
        
        c = np.mean(np.cos(x[:, 0]))
        s = np.mean(np.sin(x[:, 0]))
        
        if c >= 0:
            cr_m = np.arctan(s / c) if c != 0 else np.pi/2 * np.sign(s)
        else:
            cr_m = np.arctan(s / c) + np.pi
        
        l_m = np.mean(x[:, 1])
        
        return np.array([wrap_to_2pi(cr_m), l_m])
    
    @staticmethod
    def weighted_mean_cs(x, weights):
        """Compute weighted mean using cosine-sine representation."""
        c = np.sum(np.cos(x[:, 0]) * weights) / np.sum(weights)
        s = np.sum(np.sin(x[:, 0]) * weights) / np.sum(weights)
        r = np.sqrt(c**2 + s**2)
        
        if c >= 0:
            cr_m = np.arctan(s / c) if c != 0 else np.pi/2 * np.sign(s)
        elif c < 0:
            cr_m = np.arctan(s / c) + np.pi
        
        l_m = np.sum(x[:, 1] * weights) / np.sum(weights)
        
        return np.array([np.angle(np.exp(1j * cr_m)), l_m])
    
    @staticmethod
    def normal(x, sigma, mu, k=0):
        """Compute normal probability density with winding number k."""
        wrapped_mu = mu + np.array([2 * np.pi * k, 0])
        try:
            return multivariate_normal.pdf(x, mean=wrapped_mu, cov=sigma, allow_singular=True)
        except np.linalg.LinAlgError:
            return multivariate_normal.pdf(x, mean=wrapped_mu, cov=sigma + np.eye(2)*1e-6, allow_singular=True)
    
    @staticmethod
    def mvgmm_random(mu, sigma, p, n):
        """Generate random samples from multivariate Gaussian mixture."""
        m, d = mu.shape
        
        # Randomly pick components
        component_probs = np.cumsum(p) / np.sum(p)
        components = np.searchsorted(component_probs, np.random.rand(n))
        
        # Generate samples
        samples = np.zeros((n, d))
        for i in range(m):
            mask = components == i
            n_samples = np.sum(mask)
            if n_samples > 0:
                try:
                    samples[mask] = np.random.multivariate_normal(
                        mu[i], sigma[:, :, i], n_samples)
                except np.linalg.LinAlgError:
                    # Handle singular covariance
                    samples[mask] = mu[i] + np.random.normal(0, 0.01, (n_samples, d))
        
        return samples
    
    # Helper methods
    def _prune_small_clusters(self, cluster_centers, cluster_ids):
        """Remove clusters with too few data points."""
        if len(cluster_centers) == 0:
            return [], cluster_ids
        
        unique_ids, counts = np.unique(cluster_ids, return_counts=True)
        valid_ids = unique_ids[counts > 1]
        
        if len(valid_ids) == 0:
            return [], -np.ones_like(cluster_ids)
        
        # Keep only valid clusters
        new_centers = []
        new_ids = -np.ones_like(cluster_ids)
        
        for new_idx, old_idx in enumerate(valid_ids):
            if old_idx >= 0:  # Valid cluster ID
                new_centers.append(cluster_centers[old_idx])
                new_ids[cluster_ids == old_idx] = new_idx
        
        return new_centers, new_ids
    
    def _compute_cluster_covariances(self, cluster_ids):
        """Compute covariances for each cluster."""
        unique_ids = np.unique(cluster_ids)
        unique_ids = unique_ids[unique_ids >= 0]  # Remove invalid IDs
        
        if len(unique_ids) == 0:
            return
        
        self.clusters_means_mle = np.zeros((len(unique_ids), 2))
        self.cluster_covariance = np.zeros((2, 2, len(unique_ids)))
        
        for i, cluster_id in enumerate(unique_ids):
            cluster_data = self.data[cluster_ids == cluster_id]
            if len(cluster_data) > 1:
                mean, cov = self.mle(cluster_data)
                self.clusters_means_mle[i] = mean
                self.cluster_covariance[:, :, i] = cov
            else:
                # Single point cluster
                self.clusters_means_mle[i] = cluster_data[0]
                self.cluster_covariance[:, :, i] = np.eye(2) * 0.01
    
    def _compute_information_criteria(self):
        """Compute AIC and BIC for model selection."""
        if self.data is None or self.mean is None:
            return
        
        n_data = len(self.data)
        n_components = len(self.mean)
        n_params = n_components * (2 + 3 + 1) - 1  # mean(2) + cov(3) + weight(1) - 1
        
        # Compute log-likelihood
        log_likelihood = 0
        for i in range(n_data):
            likelihood = 0
            for j in range(n_components):
                for k in range(-self.wind, self.wind + 1):
                    wrapped_mean = self.mean[j] + np.array([2 * np.pi * k, 0])
                    try:
                        likelihood += self.p[j] * multivariate_normal.pdf(
                            self.data[i], mean=wrapped_mean, cov=self.cov[:, :, j], allow_singular=True)
                    except np.linalg.LinAlgError:
                        likelihood += self.p[j] * multivariate_normal.pdf(
                            self.data[i], mean=wrapped_mean, cov=self.cov[:, :, j] + np.eye(2)*1e-6, allow_singular=True)
            if likelihood > 0:
                log_likelihood += np.log(likelihood)
        
        # Compute criteria
        self.aic = -2 * log_likelihood + 2 * n_params
        self.bic = -2 * log_likelihood + n_params * np.log(n_data)