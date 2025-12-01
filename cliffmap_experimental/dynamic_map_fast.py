"""
High-Performance DynamicMap Implementation

Optimizations:
- Vectorized operations throughout
- Memory-efficient data structures
- Parallelized processing
- Spatial indexing for fast neighbor search
- Adaptive algorithms based on data characteristics
- GPU acceleration support
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

# Performance dependencies
HAS_NUMBA = False
HAS_CUPY = False
HAS_SKLEARN = False
HAS_SCIPY_SPATIAL = False

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
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

try:
    from sklearn.neighbors import KDTree, BallTree
    HAS_SKLEARN = True
except ImportError:
    pass

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY_SPATIAL = True
except ImportError:
    pass

from .batch_fast import BatchFast


# Numba-optimized spatial functions
@jit(nopython=True, fastmath=True, parallel=True)
def fast_spatial_grid_assignment(points, grid_size, bounds):
    """Fast assignment of points to spatial grid cells."""
    n_points = points.shape[0]
    grid_indices = np.zeros((n_points, 2), dtype=np.int32)
    
    x_min, x_max = bounds[0], bounds[1]
    y_min, y_max = bounds[2], bounds[3]
    
    cell_x_size = (x_max - x_min) / grid_size
    cell_y_size = (y_max - y_min) / grid_size
    
    for i in prange(n_points):
        grid_x = min(int((points[i, 0] - x_min) / cell_x_size), grid_size - 1)
        grid_y = min(int((points[i, 1] - y_min) / cell_y_size), grid_size - 1)
        grid_indices[i, 0] = max(0, grid_x)
        grid_indices[i, 1] = max(0, grid_y)
    
    return grid_indices


@jit(nopython=True, fastmath=True)
def fast_batch_bounds(points, batch_indices):
    """Fast computation of bounding box for batch points."""
    if len(batch_indices) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    x_min = x_max = points[batch_indices[0], 0]
    y_min = y_max = points[batch_indices[0], 1]
    
    for i in range(1, len(batch_indices)):
        idx = batch_indices[i]
        x, y = points[idx, 0], points[idx, 1]
        
        if x < x_min:
            x_min = x
        elif x > x_max:
            x_max = x
            
        if y < y_min:
            y_min = y
        elif y > y_max:
            y_max = y
    
    return np.array([x_min, x_max, y_min, y_max])


class SpatialIndex:
    """High-performance spatial indexing for fast neighbor queries."""
    
    def __init__(self, points: np.ndarray, method: str = 'auto'):
        self.points = points.astype(np.float32)
        self.method = method
        self.index = None
        
        # Choose best available spatial index
        if method == 'auto':
            if HAS_SKLEARN:
                self.method = 'kdtree'
                self.index = KDTree(self.points[:, :2])
            elif HAS_SCIPY_SPATIAL:
                self.method = 'ckdtree'
                self.index = cKDTree(self.points[:, :2])
            else:
                self.method = 'grid'
                self._build_grid_index()
        elif method == 'grid':
            self._build_grid_index()
        elif method == 'kdtree' and HAS_SKLEARN:
            self.index = KDTree(self.points[:, :2])
        elif method == 'ckdtree' and HAS_SCIPY_SPATIAL:
            self.index = cKDTree(self.points[:, :2])
        else:
            self.method = 'grid'
            self._build_grid_index()
    
    def _build_grid_index(self, grid_size: int = 50):
        """Build grid-based spatial index."""
        self.grid_size = grid_size
        
        # Compute bounds
        self.bounds = np.array([
            self.points[:, 0].min(), self.points[:, 0].max(),
            self.points[:, 1].min(), self.points[:, 1].max()
        ])
        
        # Assign points to grid cells
        self.grid_assignment = fast_spatial_grid_assignment(
            self.points[:, :2], grid_size, self.bounds
        )
        
        # Build grid lookup table
        self.grid_cells = {}
        for i, (gx, gy) in enumerate(self.grid_assignment):
            key = (gx, gy)
            if key not in self.grid_cells:
                self.grid_cells[key] = []
            self.grid_cells[key].append(i)
    
    def query_radius(self, center: np.ndarray, radius: float) -> List[int]:
        """Query points within radius of center."""
        if self.method in ['kdtree', 'ckdtree']:
            indices = self.index.query_ball_point(center[:2], radius)
            return indices if isinstance(indices, list) else indices.tolist()
        else:
            return self._grid_query_radius(center, radius)
    
    def _grid_query_radius(self, center: np.ndarray, radius: float) -> List[int]:
        """Grid-based radius query."""
        # Determine grid cells to check
        x_min, x_max = self.bounds[0], self.bounds[1]
        y_min, y_max = self.bounds[2], self.bounds[3]
        
        cell_x_size = (x_max - x_min) / self.grid_size
        cell_y_size = (y_max - y_min) / self.grid_size
        
        # Range of grid cells to check
        gx_min = max(0, int((center[0] - radius - x_min) / cell_x_size))
        gx_max = min(self.grid_size - 1, int((center[0] + radius - x_min) / cell_x_size))
        gy_min = max(0, int((center[1] - radius - y_min) / cell_y_size))
        gy_max = min(self.grid_size - 1, int((center[1] + radius - y_min) / cell_y_size))
        
        # Collect candidate points
        candidates = []
        for gx in range(gx_min, gx_max + 1):
            for gy in range(gy_min, gy_max + 1):
                if (gx, gy) in self.grid_cells:
                    candidates.extend(self.grid_cells[(gx, gy)])
        
        # Filter by actual distance
        result = []
        for idx in candidates:
            dist = np.linalg.norm(self.points[idx, :2] - center[:2])
            if dist <= radius:
                result.append(idx)
        
        return result


class AdaptiveBatchManager:
    """Adaptive batch size management based on data density and performance."""
    
    def __init__(self, initial_batch_size: int = 100):
        self.initial_batch_size = initial_batch_size
        self.performance_history = []
        self.optimal_batch_size = initial_batch_size
    
    def get_adaptive_batch_size(self, n_points: int, local_density: float) -> int:
        """Determine optimal batch size based on data characteristics."""
        # Base size on data size
        if n_points < 1000:
            base_size = 50
        elif n_points < 10000:
            base_size = 100
        else:
            base_size = 200
        
        # Adjust based on density
        density_factor = max(0.5, min(2.0, local_density / 10.0))
        adaptive_size = int(base_size * density_factor)
        
        # Use performance history if available
        if len(self.performance_history) > 3:
            best_size = min(self.performance_history, key=lambda x: x['time_per_point'])['batch_size']
            adaptive_size = int(0.7 * adaptive_size + 0.3 * best_size)
        
        return max(20, min(500, adaptive_size))  # Reasonable bounds
    
    def record_performance(self, batch_size: int, processing_time: float, n_points: int):
        """Record performance metrics for adaptive learning."""
        self.performance_history.append({
            'batch_size': batch_size,
            'processing_time': processing_time,
            'n_points': n_points,
            'time_per_point': processing_time / max(1, n_points)
        })
        
        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)


class DynamicMapFast:
    """
    High-performance CLiFF-map implementation optimized for speed.
    
    Features:
    - Vectorized operations with NumPy and Numba
    - Adaptive batch sizing
    - Spatial indexing for fast queries
    - GPU acceleration (when available)
    - Memory-efficient processing
    - Parallel execution
    """
    
    def __init__(self, 
                 batch_size: int = 100,
                 max_iterations: int = 100,
                 bandwidth: float = 0.5,
                 min_samples: int = 5,
                 parallel: bool = True,
                 n_jobs: int = -1,
                 use_gpu: bool = False,
                 spatial_index: str = 'auto',
                 adaptive_batching: bool = True,
                 verbose: bool = True):
        
        # Core parameters
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.bandwidth = bandwidth
        self.min_samples = min_samples
        self.parallel = parallel
        self.n_jobs = n_jobs if n_jobs > 0 else None
        self.use_gpu = use_gpu and HAS_CUPY
        self.adaptive_batching = adaptive_batching
        self.verbose = verbose
        
        # Performance components
        self.spatial_index = None
        self.spatial_index_method = spatial_index
        self.batch_manager = AdaptiveBatchManager(batch_size) if adaptive_batching else None
        
        # Results
        self.components = []
        self.data = None
        self.timing_info = {
            'data_loading': 0.0,
            'spatial_indexing': 0.0,
            'batch_processing': 0.0,
            'component_merging': 0.0,
            'total': 0.0
        }
        
        # Setup logging
        if verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.WARNING)
    
    def load_data(self, data: Union[np.ndarray, str], **kwargs) -> 'DynamicMapFast':
        """Fast data loading with automatic format detection."""
        start_time = time.time()
        
        if isinstance(data, str):
            # Load from file
            if data.endswith('.csv'):
                data_array = np.loadtxt(data, delimiter=',')
            else:
                data_array = np.load(data)
        else:
            data_array = np.asarray(data, dtype=np.float32)
        
        # Detect and convert format
        self.data = self._process_data_format(data_array)
        
        self.timing_info['data_loading'] = time.time() - start_time
        
        if self.verbose:
            self.logger.info(f"Loaded {len(self.data)} data points in {self.timing_info['data_loading']:.3f}s")
        
        return self
    
    def _process_data_format(self, data: np.ndarray) -> np.ndarray:
        """Fast data format processing."""
        if data.shape[1] == 4:
            # [x, y, direction, speed] -> [x, y, direction, speed]
            processed = data.copy()
            # Ensure angles are in [0, 2π]
            processed[:, 2] = np.mod(processed[:, 2], 2*np.pi)
        elif data.shape[1] == 4 and self._is_velocity_format(data):
            # [x, y, vx, vy] -> [x, y, direction, speed]
            x, y, vx, vy = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
            directions = np.arctan2(vy, vx)
            directions = np.mod(directions, 2*np.pi)  # [0, 2π]
            speeds = np.sqrt(vx**2 + vy**2)
            processed = np.column_stack([x, y, directions, speeds])
        else:
            raise ValueError(f"Unsupported data format with shape {data.shape}")
        
        return processed
    
    def _is_velocity_format(self, data: np.ndarray) -> bool:
        """Heuristic to detect velocity vs directional format."""
        # Check if third column looks like velocity component vs angle
        col3_range = data[:, 2].max() - data[:, 2].min()
        return col3_range <= 10  # Velocity components typically smaller than 2π
    
    def fit(self, data: Optional[np.ndarray] = None) -> 'DynamicMapFast':
        """High-performance CLiFF-map fitting."""
        start_time = time.time()
        
        if data is not None:
            self.load_data(data)
        
        if self.data is None:
            raise ValueError("No data available. Call load_data() first.")
        
        # Build spatial index
        self._build_spatial_index()
        
        # Process in batches
        self._process_batches_fast()
        
        # Merge and refine components
        self._merge_components()
        
        self.timing_info['total'] = time.time() - start_time
        
        if self.verbose:
            self.logger.info(f"CLiFF-map fitting completed in {self.timing_info['total']:.3f}s")
            self.logger.info(f"Found {len(self.components)} flow components")
        
        return self
    
    def _build_spatial_index(self):
        """Build spatial index for fast neighbor queries."""
        start_time = time.time()
        
        self.spatial_index = SpatialIndex(
            self.data, method=self.spatial_index_method
        )
        
        self.timing_info['spatial_indexing'] = time.time() - start_time
        
        if self.verbose:
            self.logger.info(f"Built {self.spatial_index.method} spatial index in "
                           f"{self.timing_info['spatial_indexing']:.3f}s")
    
    def _process_batches_fast(self):
        """Fast batch processing with optimizations."""
        start_time = time.time()
        
        n_points = len(self.data)
        
        # Determine batch size
        if self.batch_manager:
            local_density = self._estimate_local_density()
            actual_batch_size = self.batch_manager.get_adaptive_batch_size(
                n_points, local_density
            )
        else:
            actual_batch_size = self.batch_size
        
        # Create batches using spatial locality
        batches = self._create_spatial_batches(actual_batch_size)
        
        if self.verbose:
            self.logger.info(f"Processing {len(batches)} batches of size ~{actual_batch_size}")
        
        # Process batches
        if self.parallel and len(batches) > 1:
            batch_results = self._process_batches_parallel(batches)
        else:
            batch_results = self._process_batches_sequential(batches)
        
        # Collect components
        all_components = []
        for batch_result in batch_results:
            if batch_result is not None:
                all_components.extend(batch_result)
        
        self.components = all_components
        self.timing_info['batch_processing'] = time.time() - start_time
        
        # Record performance for adaptive learning
        if self.batch_manager:
            self.batch_manager.record_performance(
                actual_batch_size, 
                self.timing_info['batch_processing'],
                n_points
            )
    
    def _estimate_local_density(self) -> float:
        """Estimate local point density for adaptive batching."""
        n_sample = min(1000, len(self.data))
        sample_indices = np.random.choice(len(self.data), n_sample, replace=False)
        
        densities = []
        for idx in sample_indices[:100]:  # Check subset for speed
            neighbors = self.spatial_index.query_radius(
                self.data[idx], self.bandwidth * 2
            )
            densities.append(len(neighbors))
        
        return np.mean(densities) if densities else 10
    
    def _create_spatial_batches(self, batch_size: int) -> List[np.ndarray]:
        """Create spatially coherent batches for better cache locality."""
        n_points = len(self.data)
        
        if self.spatial_index.method == 'grid':
            # Use grid structure for batching
            batches = []
            used = np.zeros(n_points, dtype=bool)
            
            # Process grid cells
            for cell_indices in self.spatial_index.grid_cells.values():
                if len(cell_indices) == 0:
                    continue
                
                # Create batches from this cell
                cell_indices = [i for i in cell_indices if not used[i]]
                
                for i in range(0, len(cell_indices), batch_size):
                    batch_indices = cell_indices[i:i + batch_size]
                    if len(batch_indices) >= self.min_samples:
                        batches.append(np.array(batch_indices))
                        used[batch_indices] = True
        else:
            # Simple sequential batching
            batches = []
            for i in range(0, n_points, batch_size):
                batch_indices = np.arange(i, min(i + batch_size, n_points))
                if len(batch_indices) >= self.min_samples:
                    batches.append(batch_indices)
        
        return batches
    
    def _process_batches_parallel(self, batches: List[np.ndarray]) -> List[List[dict]]:
        """Parallel batch processing."""
        if self.use_gpu and HAS_CUPY:
            return self._process_batches_gpu(batches)
        else:
            return self._process_batches_cpu_parallel(batches)
    
    def _process_batches_cpu_parallel(self, batches: List[np.ndarray]) -> List[List[dict]]:
        """CPU parallel batch processing."""
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(self._process_single_batch_fast, batch_indices)
                for batch_indices in batches
            ]
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results.append(result)
                except Exception as e:
                    if self.verbose:
                        self.logger.warning(f"Batch processing failed: {e}")
                    results.append([])
        
        return results
    
    def _process_batches_gpu(self, batches: List[np.ndarray]) -> List[List[dict]]:
        """GPU-accelerated batch processing."""
        if not HAS_CUPY:
            return self._process_batches_cpu_parallel(batches)
        
        # Transfer data to GPU once
        data_gpu = cp.asarray(self.data)
        
        results = []
        for batch_indices in batches:
            batch_data = data_gpu[batch_indices]
            
            # Create GPU batch processor
            batch_processor = BatchFast(use_gpu=True)
            
            # Convert to angle-speed format
            batch_angular = cp.column_stack([
                batch_data[:, 2],  # direction
                batch_data[:, 3]   # speed
            ])
            
            batch_processor.set_data(cp.asnumpy(batch_angular))
            
            try:
                # Process on GPU
                centers, assignments = batch_processor.fast_mean_shift()
                batch_processor.fast_em_algorithm()
                
                # Convert results back to components
                components = []
                if len(centers) > 0:
                    for i, center in enumerate(centers):
                        # Get points assigned to this cluster
                        cluster_points = batch_data[assignments == i]
                        
                        if len(cluster_points) >= self.min_samples:
                            # Compute spatial center
                            spatial_center = cp.mean(cluster_points[:, :2], axis=0)
                            
                            component = {
                                'position': cp.asnumpy(spatial_center),
                                'direction': center[0],
                                'speed': center[1],
                                'weight': len(cluster_points) / len(batch_data),
                                'covariance': cp.asnumpy(cp.cov(cluster_points.T)),
                                'batch_id': len(results)
                            }
                            components.append(component)
                
                results.append(components)
                
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"GPU batch processing failed: {e}")
                results.append([])
        
        return results
    
    def _process_batches_sequential(self, batches: List[np.ndarray]) -> List[List[dict]]:
        """Sequential batch processing."""
        results = []
        for i, batch_indices in enumerate(batches):
            if self.verbose and i % 10 == 0:
                self.logger.info(f"Processing batch {i+1}/{len(batches)}")
            
            result = self._process_single_batch_fast(batch_indices)
            results.append(result)
        
        return results
    
    def _process_single_batch_fast(self, batch_indices: np.ndarray) -> List[dict]:
        """Fast processing of a single batch."""
        batch_data = self.data[batch_indices]
        
        if len(batch_data) < self.min_samples:
            return []
        
        try:
            # Create batch processor
            batch_processor = BatchFast(use_gpu=self.use_gpu)
            
            # Set angular data [direction, speed]
            angular_data = batch_data[:, [2, 3]]  # direction, speed columns
            batch_processor.set_data(angular_data)
            batch_processor.set_bandwidth(self.bandwidth * 0.5, self.bandwidth)
            
            # Process batch
            centers, assignments = batch_processor.fast_mean_shift()
            if len(centers) == 0:
                return []
            
            batch_processor.fast_em_algorithm()
            
            # Create components
            components = []
            for i in range(len(centers)):
                # Get points assigned to this cluster
                cluster_mask = assignments == i
                cluster_points = batch_data[cluster_mask]
                
                if len(cluster_points) >= self.min_samples:
                    # Compute spatial center
                    spatial_center = np.mean(cluster_points[:, :2], axis=0)
                    
                    # Get flow characteristics
                    direction = centers[i, 0]
                    speed = centers[i, 1]
                    
                    component = {
                        'position': spatial_center,
                        'direction': direction,
                        'speed': speed,
                        'weight': len(cluster_points) / len(batch_data),
                        'covariance': np.cov(cluster_points.T),
                        'n_points': len(cluster_points),
                        'batch_id': hash(tuple(batch_indices)) % 1000000  # Simple hash
                    }
                    components.append(component)
            
            return components
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Batch processing failed: {e}")
            return []
    
    def _merge_components(self):
        """Fast component merging."""
        start_time = time.time()
        
        if len(self.components) <= 1:
            self.timing_info['component_merging'] = time.time() - start_time
            return
        
        # Fast spatial clustering of components
        component_positions = np.array([c['position'] for c in self.components])
        
        if HAS_SKLEARN:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=self.bandwidth, min_samples=1)
            cluster_labels = clustering.fit_predict(component_positions)
        else:
            # Simple grid-based merging
            cluster_labels = self._simple_spatial_clustering(component_positions)
        
        # Merge components within clusters
        merged_components = []
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Noise in DBSCAN
                continue
                
            cluster_components = [self.components[i] for i in range(len(self.components)) 
                                if cluster_labels[i] == label]
            
            if len(cluster_components) == 1:
                merged_components.append(cluster_components[0])
            else:
                merged = self._merge_component_cluster(cluster_components)
                merged_components.append(merged)
        
        self.components = merged_components
        self.timing_info['component_merging'] = time.time() - start_time
        
        if self.verbose:
            self.logger.info(f"Merged to {len(self.components)} components in "
                           f"{self.timing_info['component_merging']:.3f}s")
    
    def _simple_spatial_clustering(self, positions: np.ndarray) -> np.ndarray:
        """Simple grid-based spatial clustering fallback."""
        n_components = len(positions)
        labels = np.arange(n_components)  # Start with each component in its own cluster
        
        # Merge nearby components
        for i in range(n_components):
            for j in range(i + 1, n_components):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < self.bandwidth and labels[i] != labels[j]:
                    # Merge clusters
                    old_label = labels[j]
                    labels[labels == old_label] = labels[i]
        
        return labels
    
    def _merge_component_cluster(self, components: List[dict]) -> dict:
        """Merge a cluster of components."""
        total_weight = sum(c['weight'] for c in components)
        
        if total_weight == 0:
            return components[0]
        
        # Weighted averages
        avg_position = np.zeros(2)
        avg_direction_x = avg_direction_y = 0.0
        avg_speed = 0.0
        total_points = 0
        
        for comp in components:
            w = comp['weight']
            avg_position += w * comp['position']
            avg_direction_x += w * np.cos(comp['direction'])
            avg_direction_y += w * np.sin(comp['direction'])
            avg_speed += w * comp['speed']
            total_points += comp.get('n_points', 1)
        
        avg_position /= total_weight
        avg_direction = np.arctan2(avg_direction_y, avg_direction_x)
        avg_speed /= total_weight
        
        return {
            'position': avg_position,
            'direction': avg_direction,
            'speed': avg_speed,
            'weight': total_weight,
            'covariance': np.eye(4),  # Simplified
            'n_points': total_points,
            'merged_from': len(components)
        }
    
    def save_xml(self, filename: str, **kwargs):
        """Fast XML export."""
        import xml.etree.ElementTree as ET
        
        root = ET.Element("cliffmap_results")
        
        # Metadata
        metadata = ET.SubElement(root, "metadata")
        ET.SubElement(metadata, "n_components").text = str(len(self.components))
        ET.SubElement(metadata, "n_datapoints").text = str(len(self.data) if self.data is not None else 0)
        ET.SubElement(metadata, "processing_time").text = f"{self.timing_info['total']:.3f}"
        
        # Performance info
        perf = ET.SubElement(metadata, "performance")
        for key, value in self.timing_info.items():
            ET.SubElement(perf, key).text = f"{value:.3f}"
        
        # Components
        components_elem = ET.SubElement(root, "components")
        for i, comp in enumerate(self.components):
            comp_elem = ET.SubElement(components_elem, "component", id=str(i))
            
            pos = ET.SubElement(comp_elem, "position")
            ET.SubElement(pos, "x").text = f"{comp['position'][0]:.6f}"
            ET.SubElement(pos, "y").text = f"{comp['position'][1]:.6f}"
            
            ET.SubElement(comp_elem, "direction").text = f"{comp['direction']:.6f}"
            ET.SubElement(comp_elem, "speed").text = f"{comp['speed']:.6f}"
            ET.SubElement(comp_elem, "weight").text = f"{comp['weight']:.6f}"
            
            if 'n_points' in comp:
                ET.SubElement(comp_elem, "n_points").text = str(comp['n_points'])
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(filename, encoding='utf-8', xml_declaration=True)
    
    def save_csv(self, filename: str, **kwargs):
        """Fast CSV export."""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            header = ['component_id', 'x', 'y', 'direction', 'speed', 'weight']
            if self.components and 'n_points' in self.components[0]:
                header.append('n_points')
            writer.writerow(header)
            
            # Data
            for i, comp in enumerate(self.components):
                row = [
                    i,
                    f"{comp['position'][0]:.6f}",
                    f"{comp['position'][1]:.6f}",
                    f"{comp['direction']:.6f}",
                    f"{comp['speed']:.6f}",
                    f"{comp['weight']:.6f}"
                ]
                
                if 'n_points' in comp:
                    row.append(comp['n_points'])
                
                writer.writerow(row)
    
    def get_performance_report(self) -> dict:
        """Get detailed performance report."""
        report = {
            'timing': self.timing_info.copy(),
            'data_info': {
                'n_points': len(self.data) if self.data is not None else 0,
                'n_components': len(self.components)
            },
            'settings': {
                'batch_size': self.batch_size,
                'bandwidth': self.bandwidth,
                'parallel': self.parallel,
                'use_gpu': self.use_gpu,
                'spatial_index': self.spatial_index.method if self.spatial_index else None,
                'adaptive_batching': self.adaptive_batching
            }
        }
        
        # Performance metrics
        if self.timing_info['total'] > 0:
            points_per_second = (len(self.data) if self.data is not None else 0) / self.timing_info['total']
            report['performance_metrics'] = {
                'points_per_second': points_per_second,
                'components_per_second': len(self.components) / self.timing_info['total']
            }
        
        return report


# Utility functions for benchmarking
def benchmark_performance(data_sizes: List[int] = None, n_trials: int = 3):
    """Comprehensive performance benchmark."""
    if data_sizes is None:
        data_sizes = [1000, 5000, 10000, 25000]
    
    results = {
        'data_sizes': data_sizes,
        'cpu_times': [],
        'gpu_times': [],
        'original_times': [],
        'speedup_vs_original': [],
        'memory_usage': []
    }
    
    # Import original for comparison
    try:
        from ..dynamic_map import DynamicMap as OriginalDynamicMap
        has_original = True
    except ImportError:
        has_original = False
        print("Original DynamicMap not available for comparison")
    
    for n_points in data_sizes:
        print(f"\nBenchmarking with {n_points} points...")
        
        # Generate test data
        np.random.seed(42)
        test_data = np.column_stack([
            np.random.uniform(-10, 10, n_points),      # x
            np.random.uniform(-5, 5, n_points),        # y
            np.random.uniform(0, 2*np.pi, n_points),   # direction
            np.random.exponential(1.0, n_points)       # speed
        ])
        
        # Test fast CPU version
        cpu_times = []
        for trial in range(n_trials):
            cliff_fast = DynamicMapFast(
                batch_size=100,
                use_gpu=False,
                adaptive_batching=True,
                verbose=False
            )
            
            start_time = time.time()
            cliff_fast.fit(test_data)
            cpu_times.append(time.time() - start_time)
        
        avg_cpu_time = np.mean(cpu_times)
        results['cpu_times'].append(avg_cpu_time)
        
        # Test GPU version if available
        if HAS_CUPY:
            gpu_times = []
            for trial in range(n_trials):
                cliff_gpu = DynamicMapFast(
                    batch_size=100,
                    use_gpu=True,
                    adaptive_batching=True,
                    verbose=False
                )
                
                start_time = time.time()
                cliff_gpu.fit(test_data)
                gpu_times.append(time.time() - start_time)
            
            avg_gpu_time = np.mean(gpu_times)
            results['gpu_times'].append(avg_gpu_time)
        else:
            results['gpu_times'].append(None)
        
        # Test original version if available
        if has_original:
            original_times = []
            for trial in range(n_trials):
                try:
                    cliff_orig = OriginalDynamicMap(
                        batch_size=100,
                        parallel=False,  # For fair comparison
                        verbose=False
                    )
                    
                    start_time = time.time()
                    cliff_orig.fit(test_data)
                    original_times.append(time.time() - start_time)
                except Exception as e:
                    print(f"Original implementation failed: {e}")
                    break
            
            if original_times:
                avg_original_time = np.mean(original_times)
                results['original_times'].append(avg_original_time)
                results['speedup_vs_original'].append(avg_original_time / avg_cpu_time)
            else:
                results['original_times'].append(None)
                results['speedup_vs_original'].append(None)
        else:
            results['original_times'].append(None)
            results['speedup_vs_original'].append(None)
        
        # Report results for this size
        print(f"  Fast CPU: {avg_cpu_time:.3f}s ({n_points/avg_cpu_time:.0f} points/s)")
        if HAS_CUPY:
            print(f"  Fast GPU: {avg_gpu_time:.3f}s ({n_points/avg_gpu_time:.0f} points/s)")
            print(f"  GPU Speedup: {avg_cpu_time/avg_gpu_time:.2f}x")
        
        if has_original and results['original_times'][-1]:
            orig_time = results['original_times'][-1]
            speedup = results['speedup_vs_original'][-1]
            print(f"  Original: {orig_time:.3f}s")
            print(f"  Fast Speedup: {speedup:.2f}x")
    
    return results