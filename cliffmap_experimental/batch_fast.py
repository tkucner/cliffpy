"""
High-Performance Batch Processing for CLiFF-map

Optimizations:
- Vectorized operations with NumPy
- Numba JIT compilation for hot loops
- Memory-efficient data structures
- Fast approximation algorithms
- GPU acceleration (when available)
"""

import numpy as np
from typing import Optional, Tuple, List
import warnings

# Performance dependencies
HAS_NUMBA = False
HAS_CUPY = False

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    def prange(x):
        return range(x)

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None


# Numba-optimized functions for hot loops
@jit(nopython=True, fastmath=True)
def fast_circular_distance(angles1, angles2):
    """Fast computation of circular distances between angles."""
    diff = angles1 - angles2
    return np.minimum(np.abs(diff), 2*np.pi - np.abs(diff))


@jit(nopython=True, fastmath=True)
def fast_euclidean_distance_2d(points1, points2):
    """Fast 2D Euclidean distance computation."""
    dx = points1[:, 0] - points2[:, 0]
    dy = points1[:, 1] - points2[:, 1]
    return np.sqrt(dx*dx + dy*dy)


@jit(nopython=True, fastmath=True, parallel=True)
def fast_gaussian_kernel(distances, bandwidth):
    """Fast Gaussian kernel evaluation."""
    result = np.zeros(distances.shape)
    inv_bandwidth = 1.0 / bandwidth
    for i in prange(distances.shape[0]):
        result[i] = np.exp(-0.5 * (distances[i] * inv_bandwidth)**2)
    return result


@jit(nopython=True, fastmath=True)
def fast_weighted_circular_mean(angles, weights):
    """Fast weighted circular mean computation."""
    if weights.sum() == 0:
        return 0.0
    
    # Convert to unit vectors
    cos_sum = np.sum(weights * np.cos(angles))
    sin_sum = np.sum(weights * np.sin(angles))
    
    return np.arctan2(sin_sum, cos_sum)


@jit(nopython=True, fastmath=True, parallel=True)
def fast_mean_shift_iteration(data_angles, data_speeds, current_center, bandwidth_angle, bandwidth_speed):
    """Single iteration of vectorized mean shift."""
    n_points = data_angles.shape[0]
    
    # Compute distances
    angle_distances = np.zeros(n_points)
    speed_distances = np.zeros(n_points)
    
    for i in prange(n_points):
        # Circular distance for angles
        angle_diff = data_angles[i] - current_center[0]
        angle_distances[i] = min(abs(angle_diff), 2*np.pi - abs(angle_diff))
        
        # Euclidean distance for speeds
        speed_distances[i] = abs(data_speeds[i] - current_center[1])
    
    # Compute weights
    weights = np.exp(-0.5 * ((angle_distances / bandwidth_angle)**2 + 
                             (speed_distances / bandwidth_speed)**2))
    
    if weights.sum() == 0:
        return current_center
    
    # Weighted means
    new_angle = fast_weighted_circular_mean(data_angles, weights)
    new_speed = np.sum(weights * data_speeds) / np.sum(weights)
    
    return np.array([new_angle, new_speed])


class BatchFast:
    """
    High-performance batch processing for CLiFF-map.
    
    Optimizations:
    - Vectorized NumPy operations
    - Numba JIT compilation
    - Memory-efficient algorithms
    - Fast approximation methods
    """
    
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu and HAS_CUPY
        self.data = None
        self.clusters_means = None
        self.cluster_ids = None
        self.mean = None
        self.cov = None
        self.p = None
        self.bandwidth_angle = 0.3
        self.bandwidth_speed = 0.3
        self.convergence_threshold = 0.01
        self.max_iterations = 50
        
        # Performance tracking
        self.timing_info = {
            'mean_shift': 0.0,
            'em_algorithm': 0.0,
            'total': 0.0
        }
    
    def set_data(self, data: np.ndarray):
        """Set input data [angle, speed] format."""
        if self.use_gpu and HAS_CUPY:
            self.data = cp.asarray(data)
        else:
            self.data = np.asarray(data, dtype=np.float32)  # Use float32 for speed
        
        # Validate data
        if self.data.shape[1] != 2:
            raise ValueError("Data must have 2 columns: [angle, speed]")
        
        # Normalize angles to [0, 2π]
        if self.use_gpu:
            self.data[:, 0] = cp.mod(self.data[:, 0], 2*np.pi)
        else:
            self.data[:, 0] = np.mod(self.data[:, 0], 2*np.pi)
    
    def set_bandwidth(self, bandwidth_angle: float, bandwidth_speed: float):
        """Set bandwidth parameters for mean shift."""
        self.bandwidth_angle = bandwidth_angle
        self.bandwidth_speed = bandwidth_speed
    
    def fast_mean_shift(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        High-performance mean shift clustering.
        
        Returns:
            cluster_centers, cluster_assignments
        """
        if self.data is None:
            raise ValueError("No data set. Call set_data() first.")
        
        import time
        start_time = time.time()
        
        if self.use_gpu and HAS_CUPY:
            return self._gpu_mean_shift()
        else:
            return self._cpu_mean_shift()
        
        self.timing_info['mean_shift'] = time.time() - start_time
    
    def _cpu_mean_shift(self) -> Tuple[np.ndarray, np.ndarray]:
        """CPU-optimized mean shift using Numba."""
        data_angles = self.data[:, 0]
        data_speeds = self.data[:, 1]
        n_points = len(self.data)
        
        # Initialize cluster centers as subset of data points
        n_initial_centers = min(n_points, max(1, n_points // 10))
        indices = np.random.choice(n_points, n_initial_centers, replace=False)
        centers = self.data[indices].copy()
        
        # Iterative refinement of centers
        for iteration in range(self.max_iterations):
            old_centers = centers.copy()
            
            # Update each center
            for i in range(len(centers)):
                centers[i] = fast_mean_shift_iteration(
                    data_angles, data_speeds, centers[i],
                    self.bandwidth_angle, self.bandwidth_speed
                )
            
            # Check convergence
            max_movement = np.max(np.linalg.norm(centers - old_centers, axis=1))
            if max_movement < self.convergence_threshold:
                break
        
        # Merge nearby centers
        centers = self._merge_close_centers(centers)
        
        # Assign points to clusters
        cluster_assignments = self._assign_to_clusters(self.data, centers)
        
        self.clusters_means = centers
        self.cluster_ids = cluster_assignments
        
        return centers, cluster_assignments
    
    def _gpu_mean_shift(self) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated mean shift using CuPy."""
        if not HAS_CUPY:
            return self._cpu_mean_shift()
        
        data_gpu = cp.asarray(self.data)
        n_points = len(data_gpu)
        
        # Initialize centers
        n_initial_centers = min(n_points, max(1, n_points // 10))
        indices = cp.random.choice(n_points, n_initial_centers, replace=False)
        centers = data_gpu[indices].copy()
        
        for iteration in range(self.max_iterations):
            old_centers = centers.copy()
            
            # Vectorized distance computation on GPU
            for i in range(len(centers)):
                # Compute distances from current center to all points
                angle_diff = data_gpu[:, 0] - centers[i, 0]
                angle_dist = cp.minimum(cp.abs(angle_diff), 2*cp.pi - cp.abs(angle_diff))
                speed_dist = cp.abs(data_gpu[:, 1] - centers[i, 1])
                
                # Gaussian weights
                weights = cp.exp(-0.5 * ((angle_dist / self.bandwidth_angle)**2 + 
                                        (speed_dist / self.bandwidth_speed)**2))
                
                if cp.sum(weights) > 0:
                    # Weighted circular mean for angle
                    cos_sum = cp.sum(weights * cp.cos(data_gpu[:, 0]))
                    sin_sum = cp.sum(weights * cp.sin(data_gpu[:, 0]))
                    new_angle = cp.arctan2(sin_sum, cos_sum)
                    
                    # Weighted mean for speed
                    new_speed = cp.sum(weights * data_gpu[:, 1]) / cp.sum(weights)
                    
                    centers[i] = cp.array([new_angle, new_speed])
            
            # Check convergence
            max_movement = cp.max(cp.linalg.norm(centers - old_centers, axis=1))
            if max_movement < self.convergence_threshold:
                break
        
        # Convert back to CPU for final processing
        centers_cpu = cp.asnumpy(centers)
        centers_cpu = self._merge_close_centers(centers_cpu)
        
        cluster_assignments = self._assign_to_clusters(cp.asnumpy(data_gpu), centers_cpu)
        
        self.clusters_means = centers_cpu
        self.cluster_ids = cluster_assignments
        
        return centers_cpu, cluster_assignments
    
    def _merge_close_centers(self, centers: np.ndarray, merge_threshold: float = 0.1) -> np.ndarray:
        """Merge centers that are too close to each other."""
        if len(centers) <= 1:
            return centers
        
        merged_centers = []
        used = np.zeros(len(centers), dtype=bool)
        
        for i, center in enumerate(centers):
            if used[i]:
                continue
            
            # Find nearby centers
            nearby = []
            for j in range(i, len(centers)):
                if used[j]:
                    continue
                
                angle_dist = fast_circular_distance(
                    np.array([center[0]]), np.array([centers[j, 0]])
                )[0]
                speed_dist = abs(center[1] - centers[j, 1])
                
                if angle_dist < merge_threshold and speed_dist < merge_threshold:
                    nearby.append(j)
                    used[j] = True
            
            if nearby:
                # Compute merged center
                nearby_centers = centers[nearby]
                # Simple average (could be weighted)
                merged_angle = fast_weighted_circular_mean(
                    nearby_centers[:, 0], np.ones(len(nearby_centers))
                )
                merged_speed = np.mean(nearby_centers[:, 1])
                merged_centers.append([merged_angle, merged_speed])
        
        return np.array(merged_centers)
    
    def _assign_to_clusters(self, data: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Assign data points to nearest cluster centers."""
        n_points = len(data)
        n_centers = len(centers)
        assignments = np.zeros(n_points, dtype=int)
        
        if n_centers == 0:
            return assignments
        
        for i in range(n_points):
            best_dist = float('inf')
            best_center = 0
            
            for j in range(n_centers):
                angle_dist = fast_circular_distance(
                    np.array([data[i, 0]]), np.array([centers[j, 0]])
                )[0]
                speed_dist = abs(data[i, 1] - centers[j, 1])
                
                total_dist = (angle_dist / self.bandwidth_angle)**2 + \
                           (speed_dist / self.bandwidth_speed)**2
                
                if total_dist < best_dist:
                    best_dist = total_dist
                    best_center = j
            
            assignments[i] = best_center
        
        return assignments
    
    def fast_em_algorithm(self, max_iterations: int = 20) -> 'BatchFast':
        """
        Fast EM algorithm for mixture model fitting.
        
        Simplified version focusing on speed over accuracy.
        """
        import time
        start_time = time.time()
        
        if self.clusters_means is None:
            raise ValueError("Run mean shift clustering first")
        
        n_clusters = len(self.clusters_means)
        n_points = len(self.data)
        
        if n_clusters == 0:
            self.mean = np.array([])
            self.cov = np.array([])
            self.p = np.array([])
            return self
        
        # Initialize parameters
        self.mean = self.clusters_means.copy()
        self.cov = np.array([np.eye(2) * 0.1 for _ in range(n_clusters)])
        self.p = np.ones(n_clusters) / n_clusters
        
        # Fast EM iterations with simplified computations
        for iteration in range(max_iterations):
            # E-step: Compute responsibilities (simplified)
            responsibilities = np.zeros((n_points, n_clusters))
            
            for k in range(n_clusters):
                # Simplified likelihood computation
                diff = self.data - self.mean[k]
                
                # Handle circular difference for angles
                diff[:, 0] = np.mod(diff[:, 0] + np.pi, 2*np.pi) - np.pi
                
                # Fast Mahalanobis distance approximation
                inv_cov = np.linalg.inv(self.cov[k] + np.eye(2) * 1e-6)
                mahal_dist = np.sum(diff @ inv_cov * diff, axis=1)
                
                responsibilities[:, k] = self.p[k] * np.exp(-0.5 * mahal_dist)
            
            # Normalize responsibilities
            row_sums = responsibilities.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            responsibilities /= row_sums
            
            # M-step: Update parameters
            for k in range(n_clusters):
                weights = responsibilities[:, k]
                weight_sum = weights.sum()
                
                if weight_sum > 1e-6:
                    # Update mixing coefficient
                    self.p[k] = weight_sum / n_points
                    
                    # Update mean (handle circular nature of angles)
                    cos_sum = np.sum(weights * np.cos(self.data[:, 0]))
                    sin_sum = np.sum(weights * np.sin(self.data[:, 0]))
                    self.mean[k, 0] = np.arctan2(sin_sum, cos_sum)
                    self.mean[k, 1] = np.sum(weights * self.data[:, 1]) / weight_sum
                    
                    # Update covariance (simplified)
                    diff = self.data - self.mean[k]
                    diff[:, 0] = np.mod(diff[:, 0] + np.pi, 2*np.pi) - np.pi
                    
                    weighted_diff = diff * np.sqrt(weights[:, None])
                    self.cov[k] = (weighted_diff.T @ weighted_diff) / weight_sum + \
                                 np.eye(2) * 1e-6
        
        self.timing_info['em_algorithm'] = time.time() - start_time
        return self
    
    def get_components(self) -> List[dict]:
        """Get final flow components."""
        if self.mean is None:
            return []
        
        components = []
        for i in range(len(self.mean)):
            component = {
                'position': np.array([0.0, 0.0]),  # Would be set by parent
                'direction': self.mean[i, 0],
                'speed': self.mean[i, 1],
                'weight': self.p[i] if self.p is not None else 1.0,
                'covariance': self.cov[i] if self.cov is not None else np.eye(2),
                'cluster_id': i
            }
            components.append(component)
        
        return components
    
    def get_timing_info(self) -> dict:
        """Get performance timing information."""
        total_time = sum(self.timing_info.values())
        self.timing_info['total'] = total_time
        return self.timing_info.copy()


# Utility function for performance testing
def benchmark_batch_performance(data_sizes: List[int], n_trials: int = 3):
    """Benchmark batch processing performance."""
    import time
    
    results = {
        'data_sizes': data_sizes,
        'cpu_times': [],
        'gpu_times': [],
        'speedup': []
    }
    
    for n_points in data_sizes:
        print(f"Benchmarking with {n_points} data points...")
        
        # Generate test data
        np.random.seed(42)
        test_data = np.column_stack([
            np.random.uniform(0, 2*np.pi, n_points),  # angles
            np.random.exponential(1.0, n_points)       # speeds
        ])
        
        # CPU timing
        cpu_times = []
        for trial in range(n_trials):
            batch_cpu = BatchFast(use_gpu=False)
            batch_cpu.set_data(test_data)
            
            start_time = time.time()
            batch_cpu.fast_mean_shift()
            batch_cpu.fast_em_algorithm()
            cpu_times.append(time.time() - start_time)
        
        avg_cpu_time = np.mean(cpu_times)
        results['cpu_times'].append(avg_cpu_time)
        
        # GPU timing (if available)
        if HAS_CUPY:
            gpu_times = []
            for trial in range(n_trials):
                batch_gpu = BatchFast(use_gpu=True)
                batch_gpu.set_data(test_data)
                
                start_time = time.time()
                batch_gpu.fast_mean_shift()
                batch_gpu.fast_em_algorithm()
                gpu_times.append(time.time() - start_time)
            
            avg_gpu_time = np.mean(gpu_times)
            results['gpu_times'].append(avg_gpu_time)
            results['speedup'].append(avg_cpu_time / avg_gpu_time)
        else:
            results['gpu_times'].append(None)
            results['speedup'].append(None)
        
        print(f"  CPU: {avg_cpu_time:.3f}s")
        if HAS_CUPY:
            print(f"  GPU: {avg_gpu_time:.3f}s (speedup: {avg_cpu_time/avg_gpu_time:.2f}x)")
    
    return results