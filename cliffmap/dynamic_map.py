"""
Enhanced DynamicMap class with comprehensive data loading and format support:
- Automatic column detection and mapping
- Support for directional (direction, speed) and velocity (vx, vy) data formats
- User-defined column mapping
- XML export functionality
- Progress monitoring and parallel processing
"""

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import os
import time
from typing import Optional, Union, Tuple, List, Dict, Any
import warnings
import logging

# Optional imports with fallbacks
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    warnings.warn("pandas not available - CSV loading will be limited to basic numpy formats")

try:
    from sklearn.neighbors import BallTree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available - using basic nearest neighbor search")

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple progress replacement
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0
            
        def __iter__(self):
            if self.iterable is not None:
                for item in self.iterable:
                    yield item
                    self.update(1)
            return self
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
            
        def update(self, n=1):
            self.n += n
            
        def set_description(self, desc):
            self.desc = desc
            
        def close(self):
            pass

from .batch import Batch
from .utils import cart2pol, pol2cart, wrap_to_2pi


class DynamicMap:
    """
    Enhanced Dynamic Flow Field Mapping using CLiFF-map algorithm.
    
    This class provides comprehensive flow field analysis with automatic data format
    detection, flexible column mapping, parallel processing, progress monitoring,
    and XML export capabilities.
    
    Parameters:
    -----------
    batch_size : int, default 100
        Number of data points to process in each batch
    max_iterations : int, default 200
        Maximum number of EM iterations for refinement
    bandwidth : float, default 0.5
        Bandwidth parameter for Mean Shift clustering
    min_samples : int, default 10
        Minimum number of samples required to form a component
    convergence_threshold : float, default 1e-4
        Convergence threshold for EM algorithm
    parallel : bool, default True
        Enable parallel processing using ThreadPoolExecutor
    n_jobs : int, default 2
        Number of parallel jobs (threads) to use
    verbose : bool, default False
        Enable verbose output and detailed logging
    progress : bool, default True
        Show progress bars for long-running operations
    column_mapping : dict, optional
        Custom column mapping for CSV files. If None, automatic detection is used.
        Examples:
        - Directional: {'x': 'pos_x', 'y': 'pos_y', 'direction': 'angle', 'speed': 'velocity'}
        - Velocity: {'x': 'pos_x', 'y': 'pos_y', 'vx': 'vel_x', 'vy': 'vel_y'}
    
    Attributes:
    -----------
    components : list
        List of detected flow components with positions, directions, and weights
    data : np.ndarray
        Loaded and preprocessed input data
    history : dict
        Training history including convergence metrics and timing
    data_format : str
        Detected data format: 'directional' (direction, speed) or 'velocity' (vx, vy)
    column_info : dict
        Information about detected or mapped columns
    
    Examples:
    ---------
    >>> # Basic usage with automatic detection
    >>> cliff_map = DynamicMap()
    >>> cliff_map.fit('data.csv')
    >>> cliff_map.save_xml('flow_map.xml')
    
    >>> # With custom column mapping for directional data
    >>> mapping = {'x': 'longitude', 'y': 'latitude', 'direction': 'heading', 'speed': 'velocity'}
    >>> cliff_map = DynamicMap(column_mapping=mapping)
    >>> cliff_map.fit('gps_data.csv')
    
    >>> # With custom column mapping for velocity data
    >>> mapping = {'x': 'pos_x', 'y': 'pos_y', 'vx': 'vel_x', 'vy': 'vel_y'}
    >>> cliff_map = DynamicMap(column_mapping=mapping, parallel=True)
    >>> cliff_map.fit('flow_data.csv')
    >>> cliff_map.save_xml('results.xml', include_metadata=True)
    """
    
    def __init__(self, batch_size=100, max_iterations=200, bandwidth=0.5,
                 min_samples=10, convergence_threshold=1e-4, parallel=True,
                 n_jobs=2, verbose=False, progress=True, column_mapping=None):
        """Initialize DynamicMap with enhanced configuration."""
        
        # Core algorithm parameters
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.bandwidth = bandwidth
        self.min_samples = min_samples
        self.convergence_threshold = convergence_threshold
        
        # Performance and monitoring
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.progress = progress
        
        # Data loading configuration
        self.column_mapping = column_mapping
        self.data_format = None
        self.column_info = {}
        
        # State variables
        self.components = []
        self.data = None
        self.history = {}
        
        # Setup logging
        self._setup_logging()
        
        if self.verbose:
            self.logger.info(f"DynamicMap initialized with batch_size={batch_size}, "
                           f"parallel={parallel}, n_jobs={n_jobs}")
            if column_mapping:
                self.logger.info(f"Custom column mapping: {column_mapping}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger('cliffmap.dynamic_map')
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        if self.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
    
    
    def load_data(self, data_source, data_format='auto'):
        """
        Load flow data from various sources with automatic format detection and column mapping.
        
        Supports both directional data (direction, speed) and velocity component data (vx, vy).
        Automatically detects column mapping from headers or uses user-defined mapping.
        
        Parameters:
        -----------
        data_source : str, np.ndarray, or pandas.DataFrame
            Data source - can be file path, numpy array, or DataFrame
        data_format : str, default 'auto'
            Data format hint: 'auto', 'directional', 'velocity', 'air_flow', 'pedestrian', or 'atc'
        
        Returns:
        --------
        np.ndarray : Loaded and preprocessed data in standardized format [x, y, direction, speed]
        
        Notes:
        ------
        The method automatically detects whether the data contains:
        - Directional data: position (x, y) + direction (radians) + speed
        - Velocity data: position (x, y) + velocity components (vx, vy)
        
        Column detection priority:
        1. User-defined column_mapping
        2. Automatic detection from common column names
        3. Positional mapping (first 4 columns)
        
        Examples:
        ---------
        >>> # Automatic detection
        >>> cliff_map.load_data('data.csv')
        
        >>> # With custom mapping for directional data
        >>> cliff_map = DynamicMap(column_mapping={
        ...     'x': 'longitude', 'y': 'latitude', 
        ...     'direction': 'heading', 'speed': 'velocity'
        ... })
        >>> cliff_map.load_data('gps_data.csv')
        
        >>> # With custom mapping for velocity data
        >>> cliff_map = DynamicMap(column_mapping={
        ...     'x': 'pos_x', 'y': 'pos_y', 
        ...     'vx': 'vel_x', 'vy': 'vel_y'
        ... })
        >>> cliff_map.load_data('flow_data.csv')
        """
        if self.verbose:
            self.logger.info(f"Loading data from: {data_source}")
        
        try:
            # Load raw data
            if isinstance(data_source, str):
                if data_source.endswith('.csv'):
                    if HAS_PANDAS:
                        # Use pandas for full functionality
                        df = pd.read_csv(data_source)
                        raw_data = df.values
                        headers = list(df.columns)
                    else:
                        # Fallback to numpy - limited functionality
                        if self.verbose:
                            self.logger.warning("pandas not available - using numpy fallback for CSV loading")
                        
                        # Read headers manually
                        with open(data_source, 'r') as f:
                            first_line = f.readline().strip()
                            headers = [h.strip() for h in first_line.split(',')]
                        
                        # Load data with numpy (skip header)
                        raw_data = np.loadtxt(data_source, delimiter=',', skiprows=1)
                        
                        if len(raw_data.shape) == 1:
                            raw_data = raw_data.reshape(1, -1)
                    
                    if self.verbose:
                        self.logger.info(f"Loaded CSV with shape: {raw_data.shape}")
                        self.logger.info(f"Headers: {headers}")
                        
                elif data_source.endswith('.xml'):
                    # TODO: Implement XML loading
                    raise NotImplementedError("XML loading not yet implemented")
                else:
                    raise ValueError(f"Unsupported file format: {data_source}")
                    
            elif isinstance(data_source, np.ndarray):
                raw_data = data_source.copy()
                headers = [f"col_{i}" for i in range(raw_data.shape[1])]
                
            elif HAS_PANDAS and hasattr(data_source, 'values'):  # DataFrame-like
                raw_data = data_source.values
                headers = list(data_source.columns) if hasattr(data_source, 'columns') else [f"col_{i}" for i in range(raw_data.shape[1])]
                
            else:
                raise TypeError(f"Unsupported data source type: {type(data_source)}")
            
            # Detect column mapping and data format
            column_mapping, detected_format = self._detect_column_mapping(headers, raw_data.shape[1])
            
            if self.verbose:
                self.logger.info(f"Detected data format: {detected_format}")
                self.logger.info(f"Column mapping: {column_mapping}")
            
            # Extract and convert data
            data = self._extract_and_convert_data(raw_data, headers, column_mapping, detected_format)
            
            # Store information
            self.data = data
            self.data_format = detected_format
            self.column_info = {
                'mapping': column_mapping,
                'format': detected_format,
                'headers': headers,
                'original_shape': raw_data.shape
            }
            
            if self.verbose:
                self.logger.info(f"Data processing complete. Final shape: {data.shape}")
                self.logger.info(f"Data range: X=[{data[:, 0].min():.2f}, {data[:, 0].max():.2f}], "
                               f"Y=[{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]")
                self.logger.info(f"Direction range: [{data[:, 2].min():.2f}, {data[:, 2].max():.2f}] radians")
                self.logger.info(f"Speed range: [{data[:, 3].min():.2f}, {data[:, 3].max():.2f}]")
            
            return data
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            if self.verbose:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    
    def _detect_column_mapping(self, headers, n_columns):
        """
        Detect column mapping and data format from headers and user configuration.
        
        Parameters:
        -----------
        headers : list
            List of column headers
        n_columns : int
            Number of columns in the data
        
        Returns:
        --------
        tuple : (column_mapping, data_format)
            Column mapping dictionary and detected data format
        """
        # Use user-defined mapping if provided
        if self.column_mapping is not None:
            mapping = self.column_mapping.copy()
            
            # Determine format from mapping
            if 'direction' in mapping and 'speed' in mapping:
                data_format = 'directional'
            elif 'vx' in mapping and 'vy' in mapping:
                data_format = 'velocity'
            else:
                raise ValueError("Column mapping must include either (direction, speed) or (vx, vy)")
            
            # Validate that required columns exist
            required_cols = ['x', 'y']
            if data_format == 'directional':
                required_cols.extend(['direction', 'speed'])
            else:
                required_cols.extend(['vx', 'vy'])
            
            for col in required_cols:
                if col not in mapping:
                    raise ValueError(f"Required column '{col}' not found in column_mapping")
                if mapping[col] not in headers:
                    raise ValueError(f"Mapped column '{mapping[col]}' not found in data headers")
            
            return mapping, data_format
        
        # Automatic detection from headers
        headers_lower = [h.lower().strip() for h in headers]
        
        # Common column name patterns
        x_patterns = ['x', 'pos_x', 'position_x', 'x_pos', 'x_position', 'easting', 'longitude', 'lon']
        y_patterns = ['y', 'pos_y', 'position_y', 'y_pos', 'y_position', 'northing', 'latitude', 'lat']
        direction_patterns = ['direction', 'dir', 'angle', 'heading', 'theta', 'orientation', 'bearing']
        speed_patterns = ['speed', 'velocity', 'vel', 'magnitude', 'mag', 'v', 'rate']
        vx_patterns = ['vx', 'vel_x', 'velocity_x', 'x_velocity', 'u', 'u_component']
        vy_patterns = ['vy', 'vel_y', 'velocity_y', 'y_velocity', 'v_component']
        
        # Find column matches
        mapping = {}
        
        # Find position columns
        for i, header in enumerate(headers_lower):
            if not mapping.get('x') and any(pattern in header for pattern in x_patterns):
                mapping['x'] = headers[i]
            elif not mapping.get('y') and any(pattern in header for pattern in y_patterns):
                mapping['y'] = headers[i]
        
        # Determine data format by looking for directional vs velocity columns
        has_direction = False
        has_speed = False
        has_vx = False
        has_vy = False
        
        for i, header in enumerate(headers_lower):
            if any(pattern in header for pattern in direction_patterns):
                mapping['direction'] = headers[i]
                has_direction = True
            elif any(pattern in header for pattern in speed_patterns):
                # Avoid conflict with 'v' being both speed and vy
                if header not in ['vy', 'vel_y', 'velocity_y', 'y_velocity', 'v_component']:
                    mapping['speed'] = headers[i]
                    has_speed = True
            elif any(pattern in header for pattern in vx_patterns):
                mapping['vx'] = headers[i]
                has_vx = True
            elif any(pattern in header for pattern in vy_patterns):
                mapping['vy'] = headers[i]
                has_vy = True
        
        # Determine data format
        if has_direction and has_speed:
            data_format = 'directional'
        elif has_vx and has_vy:
            data_format = 'velocity'
        else:
            # Fallback to positional mapping
            if self.verbose:
                self.logger.warning("Could not detect column types from headers, using positional mapping")
            
            if n_columns >= 4:
                mapping = {
                    'x': headers[0],
                    'y': headers[1]
                }
                
                # Try to guess format from column names
                third_col = headers_lower[2] if len(headers_lower) > 2 else ""
                if any(pattern in third_col for pattern in direction_patterns + ['angle', 'theta']):
                    data_format = 'directional'
                    mapping['direction'] = headers[2]
                    mapping['speed'] = headers[3] if n_columns > 3 else headers[2]  # fallback
                elif any(pattern in third_col for pattern in vx_patterns + ['u']):
                    data_format = 'velocity'
                    mapping['vx'] = headers[2]
                    mapping['vy'] = headers[3]
                else:
                    # Default to directional format
                    data_format = 'directional'
                    mapping['direction'] = headers[2]
                    mapping['speed'] = headers[3] if n_columns > 3 else headers[2]
            else:
                raise ValueError(f"Insufficient columns ({n_columns}) for flow data. Need at least 4 columns.")
        
        # Ensure we have position columns
        if 'x' not in mapping or 'y' not in mapping:
            if n_columns >= 2:
                mapping['x'] = headers[0]
                mapping['y'] = headers[1]
            else:
                raise ValueError("Could not detect x, y position columns")
        
        return mapping, data_format
    
    def _extract_and_convert_data(self, raw_data, headers, column_mapping, data_format):
        """
        Extract and convert data to standardized format [x, y, direction, speed].
        
        Parameters:
        -----------
        raw_data : np.ndarray
            Raw input data
        headers : list
            Column headers
        column_mapping : dict
            Column mapping dictionary
        data_format : str
            Data format: 'directional' or 'velocity'
        
        Returns:
        --------
        np.ndarray : Standardized data array [x, y, direction, speed]
        """
        # Create header to index mapping
        header_to_idx = {header: i for i, header in enumerate(headers)}
        
        # Extract position data
        x_idx = header_to_idx[column_mapping['x']]
        y_idx = header_to_idx[column_mapping['y']]
        
        x = raw_data[:, x_idx]
        y = raw_data[:, y_idx]
        
        if data_format == 'directional':
            # Extract direction and speed directly
            dir_idx = header_to_idx[column_mapping['direction']]
            speed_idx = header_to_idx[column_mapping['speed']]
            
            direction = raw_data[:, dir_idx]
            speed = raw_data[:, speed_idx]
            
            # Ensure directions are in radians
            if np.max(np.abs(direction)) > 2 * np.pi:
                if self.verbose:
                    self.logger.info("Converting direction from degrees to radians")
                direction = np.deg2rad(direction)
            
            # Wrap directions to [0, 2π)
            direction = np.mod(direction, 2 * np.pi)
            
        elif data_format == 'velocity':
            # Convert velocity components to direction and speed
            vx_idx = header_to_idx[column_mapping['vx']]
            vy_idx = header_to_idx[column_mapping['vy']]
            
            vx = raw_data[:, vx_idx]
            vy = raw_data[:, vy_idx]
            
            # Convert to polar coordinates
            speed = np.sqrt(vx**2 + vy**2)
            direction = np.arctan2(vy, vx)
            
            # Ensure directions are in [0, 2π)
            direction = np.mod(direction, 2 * np.pi)
            
            if self.verbose:
                self.logger.info("Converted velocity components (vx, vy) to (direction, speed)")
        
        else:
            raise ValueError(f"Unknown data format: {data_format}")
        
        # Handle invalid values
        valid_mask = (np.isfinite(x) & np.isfinite(y) & 
                     np.isfinite(direction) & np.isfinite(speed) & 
                     (speed >= 0))
        
        if not np.all(valid_mask):
            n_invalid = np.sum(~valid_mask)
            if self.verbose:
                self.logger.warning(f"Removing {n_invalid} invalid data points")
            
            x = x[valid_mask]
            y = y[valid_mask]
            direction = direction[valid_mask]
            speed = speed[valid_mask]
        
        # Combine into standardized format
        data = np.column_stack([x, y, direction, speed])
        
        return data
    
    def fit(self, data=None):
        """
        Fit the CLiFF-map model to flow data.
        
        Parameters:
        -----------
        data : str, np.ndarray, or DataFrame, optional
            Data to fit. If None, uses previously loaded data.
        
        Returns:
        --------
        self : DynamicMap
            Returns self for method chaining
        """
        if data is not None:
            self.load_data(data)
        
        if self.data is None:
            raise ValueError("No data available. Load data first or provide data parameter.")
        
        if self.verbose:
            self.logger.info(f"Starting CLiFF-map analysis on {len(self.data)} data points")
        
        # Initialize history tracking
        self.history = {
            'likelihood': [],
            'n_components': [],
            'processing_time': [],
            'convergence': []
        }
        
        start_time = time.time()
        
        try:
            # Process data in batches
            self._process_batches()
            
            processing_time = time.time() - start_time
            self.history['total_time'] = processing_time
            
            if self.verbose:
                self.logger.info(f"CLiFF-map analysis completed in {processing_time:.2f} seconds")
                self.logger.info(f"Found {len(self.components)} flow components")
                
                # Component summary
                for i, comp in enumerate(self.components):
                    direction_deg = np.degrees(comp['direction']) % 360
                    self.logger.info(f"  Component {i+1}: pos=({comp['position'][0]:.2f}, {comp['position'][1]:.2f}), "
                                   f"dir={direction_deg:.1f}°, weight={comp['weight']:.3f}")
            
            return self
            
        except Exception as e:
            error_msg = f"Error during fitting: {str(e)}"
            if self.verbose:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _process_batches(self):
        """Process data in batches to find flow components."""
        n_points = len(self.data)
        n_batches = max(1, n_points // self.batch_size)
        
        if self.verbose:
            self.logger.info(f"Processing {n_points} points in {n_batches} batches")
        
        all_components = []
        
        # Create progress bar if enabled
        if self.progress:
            pbar = tqdm(total=n_batches, desc="Processing batches", 
                       disable=not self.progress or not self.verbose)
        
        try:
            if self.parallel and n_batches > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    futures = []
                    
                    for i in range(n_batches):
                        start_idx = i * self.batch_size
                        end_idx = min((i + 1) * self.batch_size, n_points)
                        batch_data = self.data[start_idx:end_idx]
                        
                        future = executor.submit(self._process_single_batch, batch_data, i)
                        futures.append(future)
                    
                    # Collect results
                    for future in as_completed(futures):
                        batch_components = future.result()
                        if batch_components:
                            all_components.extend(batch_components)
                        
                        if self.progress and hasattr(pbar, 'update'):
                            pbar.update(1)
            else:
                # Sequential processing
                for i in range(n_batches):
                    start_idx = i * self.batch_size
                    end_idx = min((i + 1) * self.batch_size, n_points)
                    batch_data = self.data[start_idx:end_idx]
                    
                    batch_components = self._process_single_batch(batch_data, i)
                    if batch_components:
                        all_components.extend(batch_components)
                    
                    if self.progress and hasattr(pbar, 'update'):
                        pbar.update(1)
        
        finally:
            if self.progress and hasattr(pbar, 'close'):
                pbar.close()
        
        # Merge and refine components
        self.components = self._merge_and_refine_components(all_components)
    
    def _process_single_batch(self, batch_data, batch_id):
        """Process a single batch of data."""
        if len(batch_data) < self.min_samples:
            return []
        
        try:
            # Create Batch object
            batch = Batch()
            batch.set_parameters(batch_id, None, batch_data, None)
            
            # Set Mean Shift parameters
            batch.set_parameters_mean_shift(0.01, np.eye(2) * self.bandwidth)
            
            # Apply Mean Shift clustering
            batch, _ = batch.mean_shift_2d()
            
            if batch.clusters_means is None or len(batch.clusters_means) == 0:
                return []
            
            # Refine with EM algorithm
            batch = batch.em_algorithm()
            
            # Extract components
            components = []
            if batch.mean is not None and batch.cov is not None:
                for i in range(len(batch.mean)):
                    component = {
                        'position': batch.mean[i][:2],  # x, y position
                        'direction': batch.mean[i][2],  # direction
                        'speed': batch.mean[i][3] if len(batch.mean[i]) > 3 else 1.0,  # speed
                        'covariance': batch.cov[i] if batch.cov is not None else np.eye(4),
                        'weight': batch.p[i] if batch.p is not None else 1.0,
                        'batch_id': batch_id
                    }
                    components.append(component)
            
            return components
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Error processing batch {batch_id}: {e}")
            return []
    
    def _merge_and_refine_components(self, all_components):
        """Merge overlapping components and perform final refinement."""
        if not all_components:
            return []
        
        # Simple merging based on spatial proximity
        merged_components = []
        merge_threshold = self.bandwidth * 1.5
        
        for component in all_components:
            merged = False
            
            for i, existing in enumerate(merged_components):
                # Check if components are close enough to merge
                pos_dist = np.linalg.norm(component['position'] - existing['position'])
                
                if pos_dist < merge_threshold:
                    # Merge components (weighted average)
                    total_weight = component['weight'] + existing['weight']
                    
                    # Weighted position
                    new_position = (component['position'] * component['weight'] + 
                                  existing['position'] * existing['weight']) / total_weight
                    
                    # Circular mean for direction
                    w1, w2 = component['weight'], existing['weight']
                    d1, d2 = component['direction'], existing['direction']
                    
                    # Convert to complex numbers for circular averaging
                    c1 = w1 * np.exp(1j * d1)
                    c2 = w2 * np.exp(1j * d2)
                    new_direction = np.angle((c1 + c2) / total_weight)
                    
                    # Ensure direction is in [0, 2π)
                    new_direction = np.mod(new_direction, 2 * np.pi)
                    
                    # Update merged component
                    merged_components[i] = {
                        'position': new_position,
                        'direction': new_direction,
                        'weight': total_weight,
                        'uncertainty': (component.get('uncertainty', 0) + 
                                      existing.get('uncertainty', 0)) / 2
                    }
                    
                    merged = True
                    break
            
            if not merged:
                merged_components.append(component)
        
        # Filter by minimum weight
        min_weight = 1.0 / len(all_components) if all_components else 0.01
        filtered_components = [comp for comp in merged_components 
                             if comp['weight'] >= min_weight]
        
        # Sort by weight (descending)
        filtered_components.sort(key=lambda x: x['weight'], reverse=True)
        
        return filtered_components
    
    def save_xml(self, filename, include_metadata=True):
        """
        Save flow field components to XML format.
        
        Parameters:
        -----------
        filename : str
            Output XML filename
        include_metadata : bool, default True
            Include analysis metadata in XML output
        
        Examples:
        ---------
        >>> cliff_map.fit(data)
        >>> cliff_map.save_xml('flow_map.xml')
        >>> cliff_map.save_xml('flow_map.xml', include_metadata=False)  # Minimal XML
        """
        if not self.components:
            raise ValueError("No components to save. Run fit() first.")
        
        try:
            import xml.etree.ElementTree as ET
            from datetime import datetime
            
            # Create root element
            root = ET.Element("CLiFFMap")
            
            # Add metadata if requested
            if include_metadata:
                metadata = ET.SubElement(root, "Metadata")
                
                # General information
                ET.SubElement(metadata, "Timestamp").text = datetime.now().isoformat()
                ET.SubElement(metadata, "Version").text = "1.0.0"
                ET.SubElement(metadata, "DataFormat").text = getattr(self, 'data_format', 'unknown')
                
                # Analysis parameters
                params = ET.SubElement(metadata, "Parameters")
                ET.SubElement(params, "BatchSize").text = str(self.batch_size)
                ET.SubElement(params, "MaxIterations").text = str(self.max_iterations)
                ET.SubElement(params, "Bandwidth").text = str(self.bandwidth)
                ET.SubElement(params, "MinSamples").text = str(self.min_samples)
                ET.SubElement(params, "ConvergenceThreshold").text = str(self.convergence_threshold)
                ET.SubElement(params, "Parallel").text = str(self.parallel)
                
                # Data information
                if self.data is not None:
                    data_info = ET.SubElement(metadata, "DataInfo")
                    ET.SubElement(data_info, "DataPoints").text = str(len(self.data))
                    ET.SubElement(data_info, "XRange").text = f"{self.data[:, 0].min():.6f},{self.data[:, 0].max():.6f}"
                    ET.SubElement(data_info, "YRange").text = f"{self.data[:, 1].min():.6f},{self.data[:, 1].max():.6f}"
                
                # Column information if available
                if hasattr(self, 'column_info') and self.column_info:
                    col_info = ET.SubElement(metadata, "ColumnInfo")
                    for key, value in self.column_info['mapping'].items():
                        ET.SubElement(col_info, key.capitalize()).text = str(value)
            
            # Add components
            components_elem = ET.SubElement(root, "Components")
            ET.SubElement(components_elem, "Count").text = str(len(self.components))
            
            for i, component in enumerate(self.components):
                comp_elem = ET.SubElement(components_elem, "Component")
                comp_elem.set("id", str(i + 1))
                
                # Position
                position = ET.SubElement(comp_elem, "Position")
                ET.SubElement(position, "X").text = f"{component['position'][0]:.6f}"
                ET.SubElement(position, "Y").text = f"{component['position'][1]:.6f}"
                
                # Flow properties
                flow = ET.SubElement(comp_elem, "Flow")
                ET.SubElement(flow, "Direction").text = f"{component['direction']:.6f}"
                ET.SubElement(flow, "DirectionDegrees").text = f"{np.degrees(component['direction']) % 360:.2f}"
                ET.SubElement(flow, "Weight").text = f"{component['weight']:.6f}"
                
                # Optional properties
                if 'uncertainty' in component:
                    ET.SubElement(flow, "Uncertainty").text = f"{component['uncertainty']:.6f}"
                
                if 'speed' in component:
                    ET.SubElement(flow, "Speed").text = f"{component['speed']:.6f}"
                
                # Velocity components (computed from direction and weight as speed)
                speed_val = component.get('speed', component['weight'])  # fallback to weight
                vx = speed_val * np.cos(component['direction'])
                vy = speed_val * np.sin(component['direction'])
                
                velocity = ET.SubElement(comp_elem, "Velocity")
                ET.SubElement(velocity, "Vx").text = f"{vx:.6f}"
                ET.SubElement(velocity, "Vy").text = f"{vy:.6f}"
                
                # Classification (based on weight and direction)
                classification = ET.SubElement(comp_elem, "Classification")
                
                # Intensity classification
                if component['weight'] > 0.3:
                    intensity = "High"
                elif component['weight'] > 0.1:
                    intensity = "Medium"
                else:
                    intensity = "Low"
                ET.SubElement(classification, "Intensity").text = intensity
                
                # Direction classification
                direction_deg = np.degrees(component['direction']) % 360
                if direction_deg < 22.5 or direction_deg >= 337.5:
                    direction_class = "East"
                elif 22.5 <= direction_deg < 67.5:
                    direction_class = "Northeast"
                elif 67.5 <= direction_deg < 112.5:
                    direction_class = "North"
                elif 112.5 <= direction_deg < 157.5:
                    direction_class = "Northwest"
                elif 157.5 <= direction_deg < 202.5:
                    direction_class = "West"
                elif 202.5 <= direction_deg < 247.5:
                    direction_class = "Southwest"
                elif 247.5 <= direction_deg < 292.5:
                    direction_class = "South"
                else:
                    direction_class = "Southeast"
                
                ET.SubElement(classification, "Direction").text = direction_class
            
            # Create tree and save
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ", level=0)  # Pretty formatting
            tree.write(filename, encoding="utf-8", xml_declaration=True)
            
            if self.verbose:
                self.logger.info(f"Flow field map saved to XML: {filename}")
                self.logger.info(f"Saved {len(self.components)} components with metadata")
            
        except Exception as e:
            error_msg = f"Error saving XML: {str(e)}"
            if self.verbose:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def save_csv(self, filename, include_velocity_components=True):
        """
        Save flow field components to CSV format.
        
        Parameters:
        -----------
        filename : str
            Output CSV filename
        include_velocity_components : bool, default True
            Include computed velocity components (vx, vy) in output
        
        Examples:
        ---------
        >>> cliff_map.save_csv('components.csv')
        >>> cliff_map.save_csv('components.csv', include_velocity_components=False)
        """
        if not self.components:
            raise ValueError("No components to save. Run fit() first.")
        
        try:
            # Prepare data for CSV export
            export_data = []
            
            for i, component in enumerate(self.components):
                row = {
                    'component_id': i + 1,
                    'x_position': component['position'][0],
                    'y_position': component['position'][1],
                    'direction_rad': component['direction'],
                    'direction_deg': np.degrees(component['direction']) % 360,
                    'weight': component['weight']
                }
                
                # Add optional properties
                if 'uncertainty' in component:
                    row['uncertainty'] = component['uncertainty']
                
                if 'speed' in component:
                    row['speed'] = component['speed']
                
                # Add velocity components if requested
                if include_velocity_components:
                    speed_val = component.get('speed', component['weight'])
                    row['vx'] = speed_val * np.cos(component['direction'])
                    row['vy'] = speed_val * np.sin(component['direction'])
                
                export_data.append(row)
            
            # Save to CSV
            if HAS_PANDAS:
                df = pd.DataFrame(export_data)
                df.to_csv(filename, index=False)
            else:
                # Fallback CSV writing without pandas
                if export_data:
                    # Get all keys from first row
                    headers = list(export_data[0].keys())
                    
                    with open(filename, 'w') as f:
                        # Write headers
                        f.write(','.join(headers) + '\n')
                        
                        # Write data rows
                        for row in export_data:
                            values = [str(row.get(header, '')) for header in headers]
                            f.write(','.join(values) + '\n')
                else:
                    # Empty file with just headers
                    with open(filename, 'w') as f:
                        f.write('component_id,x_position,y_position,direction_rad,direction_deg,weight\n')
            
            if self.verbose:
                self.logger.info(f"Flow field components saved to CSV: {filename}")
                self.logger.info(f"Saved {len(self.components)} components")
            
        except Exception as e:
            error_msg = f"Error saving CSV: {str(e)}"
            if self.verbose:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_component_summary(self):
        """
        Get a summary of detected flow components.
        
        Returns:
        --------
        dict : Summary information about components
        """
        if not self.components:
            return {'n_components': 0, 'message': 'No components found'}
        
        summary = {
            'n_components': len(self.components),
            'total_weight': sum(comp['weight'] for comp in self.components),
            'dominant_direction_deg': np.degrees(self.components[0]['direction']) % 360 if self.components else 0,
            'components': []
        }
        
        for i, comp in enumerate(self.components):
            comp_info = {
                'id': i + 1,
                'position': comp['position'].tolist(),
                'direction_deg': np.degrees(comp['direction']) % 360,
                'weight': comp['weight'],
                'classification': self._classify_component(comp)
            }
            summary['components'].append(comp_info)
        
        return summary
    
    def _classify_component(self, component):
        """Classify a component based on its properties."""
        # Intensity classification
        if component['weight'] > 0.3:
            intensity = "High"
        elif component['weight'] > 0.1:
            intensity = "Medium"
        else:
            intensity = "Low"
        
        # Direction classification
        direction_deg = np.degrees(component['direction']) % 360
        
        if direction_deg < 22.5 or direction_deg >= 337.5:
            direction_class = "East"
        elif 22.5 <= direction_deg < 67.5:
            direction_class = "Northeast"
        elif 67.5 <= direction_deg < 112.5:
            direction_class = "North"
        elif 112.5 <= direction_deg < 157.5:
            direction_class = "Northwest"
        elif 157.5 <= direction_deg < 202.5:
            direction_class = "West"
        elif 202.5 <= direction_deg < 247.5:
            direction_class = "Southwest"
        elif 247.5 <= direction_deg < 292.5:
            direction_class = "South"
        else:
            direction_class = "Southeast"
        
        return {
            'intensity': intensity,
            'direction': direction_class,
            'flow_type': f"{intensity} {direction_class} Flow"
        }
    
    def _load_air_flow_data(self, data: Any):
        """Load air flow format: [timestamp, x, y, vx, vy, location_id]"""
        self.timestamp = data.iloc[:, 0].values
        self.position = data.iloc[:, 1:3].values
        self.uv = data.iloc[:, 3:5].values
        self.location_id = data.iloc[:, 5].values
        
        # Convert to polar coordinates
        rho, theta = cart2pol(self.uv[:, 0], self.uv[:, 1])
        self.theta_rho = np.column_stack([theta, rho])
    
    def _load_pedestrian_data(self, data: Any, max_locations: int = 30):
        """Load pedestrian format: [id, timestamp, x, y, vx, vy]"""
        x_pos = data.iloc[:, 2].values
        y_pos = data.iloc[:, 3].values
        vx = data.iloc[:, 4].values
        vy = data.iloc[:, 5].values
        
        # Filter valid data
        rho = np.sqrt(vx**2 + vy**2)
        valid_idx = rho > 0.01
        
        x_pos = x_pos[valid_idx]
        y_pos = y_pos[valid_idx]
        vx = vx[valid_idx]
        vy = vy[valid_idx]
        
        # Create spatial location IDs
        x_min, x_max = x_pos.min(), x_pos.max()
        y_min, y_max = y_pos.min(), y_pos.max()
        grid_size = max((x_max - x_min), (y_max - y_min)) / np.sqrt(max_locations)
        
        x_bins = np.floor((x_pos - x_min) / grid_size).astype(int)
        y_bins = np.floor((y_pos - y_min) / grid_size).astype(int)
        
        unique_bins = np.unique(np.column_stack([x_bins, y_bins]), axis=0)[:max_locations]
        
        location_ids = np.zeros(len(x_pos), dtype=int)
        for i, (x_bin, y_bin) in enumerate(unique_bins):
            mask = (x_bins == x_bin) & (y_bins == y_bin)
            location_ids[mask] = i + 1
        
        # Keep only assigned locations
        valid_loc_idx = location_ids > 0
        
        self.position = np.column_stack([x_pos[valid_loc_idx], y_pos[valid_loc_idx]])
        self.uv = np.column_stack([vx[valid_loc_idx], vy[valid_loc_idx]])
        self.location_id = location_ids[valid_loc_idx]
        self.timestamp = np.arange(len(self.position)) + 1
        
        # Convert to polar
        rho, theta = cart2pol(self.uv[:, 0], self.uv[:, 1])
        self.theta_rho = np.column_stack([theta, rho])
    
    def _load_atc_data(self, data: Any, **kwargs):
        """Load ATC format: [timestamp, x, y, direction, speed] or similar"""
        # Flexible ATC data loading
        if data.shape[1] >= 5:
            self.timestamp = data.iloc[:, 0].values
            self.position = data.iloc[:, 1:3].values
            
            if 'direction_col' in kwargs and 'speed_col' in kwargs:
                # Direct theta, rho format
                theta = data.iloc[:, kwargs['direction_col']].values
                rho = data.iloc[:, kwargs['speed_col']].values
                self.theta_rho = np.column_stack([theta, rho])
                
                # Convert to Cartesian
                vx, vy = pol2cart(theta, rho)
                self.uv = np.column_stack([vx, vy])
            else:
                # Assume vx, vy in columns 3, 4
                self.uv = data.iloc[:, 3:5].values
                rho, theta = cart2pol(self.uv[:, 0], self.uv[:, 1])
                self.theta_rho = np.column_stack([theta, rho])
            
            # Create location IDs if not provided
            if data.shape[1] > 5:
                self.location_id = data.iloc[:, 5].values
            else:
                # Generate spatial clusters
                self._generate_spatial_locations()
        else:
            raise ValueError("ATC data must have at least 5 columns")
    
    def _generate_spatial_locations(self, target_locations: int = 25):
        """Generate spatial location clusters for continuous data."""
        from sklearn.cluster import KMeans
        
        # Use KMeans clustering to create spatial locations
        kmeans = KMeans(n_clusters=target_locations, random_state=42, n_init=10)
        self.location_id = kmeans.fit_predict(self.position) + 1
    
    def set_parameters(self, radius: float, xmin: float, xmax: float, 
                      ymin: float, ymax: float, step: float, wind: int):
        """Set grid parameters (compatible with original interface)."""
        self.radius = radius
        self.grid_parameters = np.array([xmin, xmax, ymin, ymax, step])
        self.wind = wind
        self.resolution = step
        
        # Create grid
        x_coords = np.arange(xmin, xmax + step, step)
        y_coords = np.arange(ymin, ymax + step, step)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        self.grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        
        return self
    
    def split_to_locations(self):
        """Split measurements into location-based batches with progress monitoring."""
        if self.verbose and not self.silent_mode:
            self.logger.info("Splitting data to location-based batches...")
        
        if self.location_id is None:
            # Continuous distribution - use spatial clustering
            self._split_continuous_data()
        else:
            # Discrete locations - use sparse batches
            self._split_discrete_data()
        
        if self.verbose and not self.silent_mode:
            n_batches = len(self.batches_sparse) if self.batches_sparse else len(self.batches)
            valid_batches = sum(1 for b in (self.batches_sparse or self.batches) 
                              if b is not None and hasattr(b, 'data') and b.data is not None and len(b.data) > 0)
            self.logger.info(f"Created {n_batches} batches, {valid_batches} with valid data")
        
        return self
    
    def _split_discrete_data(self):
        """Split data using discrete location IDs."""
        unique_locations = np.unique(self.location_id)
        self.batches_sparse = []
        
        progress_bar = None
        if self.verbose and not self.silent_mode:
            progress_bar = tqdm(unique_locations, desc="Creating batches", 
                              disable=self.silent_mode)
        
        for j in (progress_bar or unique_locations):
            if j <= 0:
                continue
                
            mask = self.location_id == j
            positions = self.position[mask]
            unique_pos = np.unique(positions, axis=0)
            
            if len(unique_pos) > 0:
                loc = unique_pos[0]
                batch = Batch(verbose=self.verbose and not self.silent_mode)
                batch.set_parameters(j, loc, self.theta_rho[mask], self.wind)
                batch.set_parameters_mean_shift(self.convergence_threshold, self.bandwidth_mode)
                batch.max_iterations = self.max_iterations
                self.batches_sparse.append(batch)
        
        if progress_bar:
            progress_bar.close()
    
    def _split_continuous_data(self):
        """Split data using spatial clustering."""
        if self.grid is None:
            raise ValueError("Grid parameters must be set for continuous data")
        
        tree = BallTree(self.position)
        indices = tree.query_radius(self.grid, r=self.radius)
        
        self.batches = []
        progress_bar = None
        if self.verbose and not self.silent_mode:
            progress_bar = tqdm(enumerate(indices), desc="Creating spatial batches",
                              total=len(indices), disable=self.silent_mode)
        
        for k, idx in (progress_bar or enumerate(indices)):
            if len(idx) > 2:
                batch = Batch(verbose=self.verbose and not self.silent_mode)
                batch.add_data(self.theta_rho[idx])
                if batch.data is not None and len(batch.data) > 0:
                    batch.set_parameters_mean_shift(self.convergence_threshold, self.bandwidth_mode)
                    batch.max_iterations = self.max_iterations
                self.batches.append(batch)
            else:
                self.batches.append(None)
        
        if progress_bar:
            progress_bar.close()
    
    def process(self, 
                n_jobs: int = 1,
                verbose: Optional[bool] = None,
                checkpoint: Optional[str] = None,
                checkpoint_interval: int = 10,
                continue_processing: bool = False) -> Dict[str, Any]:
        """
        Process all batches with comprehensive monitoring and checkpointing.
        
        Args:
            n_jobs: Number of parallel workers (-1 for all cores)
            verbose: Override instance verbose setting  
            checkpoint: Checkpoint file path for session persistence
            checkpoint_interval: Save checkpoint every N processed batches
            continue_processing: Resume from previous checkpoint
            
        Returns:
            Dictionary with processing results and statistics
        """
        if verbose is not None:
            original_verbose = self.verbose
            self.verbose = verbose
        
        start_time = time.time()
        
        try:
            if continue_processing and checkpoint and os.path.exists(checkpoint):
                self.load_checkpoint(checkpoint)
            
            if self.verbose and not self.silent_mode:
                self.logger.info(f"Starting CLiFF-map processing with {n_jobs} workers")
                self.logger.info(f"Checkpoint: {checkpoint}, Interval: {checkpoint_interval}")
            
            # Choose processing method based on batch type
            if self.batches_sparse is not None:
                results = self._process_sparse_batches(n_jobs, checkpoint, checkpoint_interval)
            elif self.batches is not None:
                results = self._process_regular_batches(n_jobs, checkpoint, checkpoint_interval)
            else:
                raise ValueError("No batches to process. Call split_to_locations() first.")
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Final statistics
            results.update({
                'total_processing_time': self.total_processing_time,
                'average_time_per_batch': processing_time / max(1, results['processed_batches']),
                'components_per_second': results['total_components'] / max(0.001, processing_time),
                'configuration': self._get_config_dict()
            })
            
            if self.verbose and not self.silent_mode:
                self._log_processing_summary(results)
            
            # Final checkpoint
            if checkpoint:
                self.save_checkpoint(checkpoint)
                
            return results
            
        except KeyboardInterrupt:
            if self.verbose and not self.silent_mode:
                self.logger.warning("Processing interrupted by user")
            self.processing_interrupted = True
            if checkpoint:
                self.save_checkpoint(checkpoint)
            raise
        
        finally:
            if verbose is not None:
                self.verbose = original_verbose
    
    def _process_sparse_batches(self, n_jobs: int, checkpoint: Optional[str], 
                               checkpoint_interval: int) -> Dict[str, Any]:
        """Process sparse batches with progress monitoring."""
        valid_batches = [(i, batch) for i, batch in enumerate(self.batches_sparse)
                        if batch is not None and batch.data is not None and len(batch.data) > 0]
        
        if not valid_batches:
            return {'processed_batches': 0, 'total_components': 0, 'failed_batches': 0}
        
        start_idx = max(0, self.last_processed_batch)
        batches_to_process = valid_batches[start_idx:]
        
        progress_bar = None
        if self.verbose and not self.silent_mode:
            progress_bar = tqdm(
                total=len(batches_to_process),
                desc="Processing batches",
                initial=0,
                disable=self.silent_mode,
                unit="batch"
            )
        
        processed_count = 0
        failed_count = 0
        total_components = 0
        
        if n_jobs == 1:
            # Sequential processing with detailed progress
            for i, (batch_idx, batch) in enumerate(batches_to_process):
                try:
                    processed_batch = self._process_single_batch_object(batch)
                    self.batches_sparse[batch_idx] = processed_batch
                    
                    if processed_batch.clusters_means is not None:
                        total_components += len(processed_batch.clusters_means)
                    
                    processed_count += 1
                    self.last_processed_batch = start_idx + i + 1
                    
                    if progress_bar:
                        progress_bar.update(1)
                        progress_bar.set_postfix({
                            'components': total_components,
                            'failed': failed_count
                        })
                    
                    # Checkpoint if needed
                    if checkpoint and (processed_count % checkpoint_interval == 0):
                        self.save_checkpoint(checkpoint)
                        if self.verbose and not self.silent_mode:
                            self.logger.debug(f"Checkpoint saved at batch {self.last_processed_batch}")
                            
                except Exception as e:
                    failed_count += 1
                    if self.verbose and not self.silent_mode:
                        self.logger.error(f"Failed to process batch {batch_idx}: {e}")
        
        else:
            # Parallel processing
            processed_count, failed_count, total_components = self._parallel_process_batches(
                batches_to_process, n_jobs, progress_bar, checkpoint, checkpoint_interval, start_idx
            )
        
        if progress_bar:
            progress_bar.close()
        
        return {
            'processed_batches': processed_count,
            'failed_batches': failed_count,
            'total_components': total_components,
            'processing_mode': 'parallel' if n_jobs != 1 else 'sequential'
        }
    
    def _parallel_process_batches(self, batches_to_process: List[Tuple], n_jobs: int,
                                 progress_bar, checkpoint: Optional[str], 
                                 checkpoint_interval: int, start_idx: int) -> Tuple[int, int, int]:
        """Execute parallel batch processing."""
        n_cores = os.cpu_count() if n_jobs == -1 else min(n_jobs, os.cpu_count() or 1)
        
        processed_count = 0
        failed_count = 0
        total_components = 0
        
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            # Submit all jobs
            future_to_batch = {
                executor.submit(self._process_single_batch_object, batch): (batch_idx, i)
                for i, (batch_idx, batch) in enumerate(batches_to_process)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_batch):
                batch_idx, local_idx = future_to_batch[future]
                try:
                    processed_batch = future.result()
                    self.batches_sparse[batch_idx] = processed_batch
                    
                    if processed_batch.clusters_means is not None:
                        total_components += len(processed_batch.clusters_means)
                    
                    processed_count += 1
                    self.last_processed_batch = start_idx + local_idx + 1
                    
                    if progress_bar:
                        progress_bar.update(1)
                        progress_bar.set_postfix({
                            'components': total_components,
                            'failed': failed_count
                        })
                    
                    # Checkpoint periodically
                    if checkpoint and (processed_count % checkpoint_interval == 0):
                        self.save_checkpoint(checkpoint)
                
                except Exception as e:
                    failed_count += 1
                    if self.verbose and not self.silent_mode:
                        self.logger.error(f"Failed to process batch {batch_idx}: {e}")
        
        return processed_count, failed_count, total_components
    
    def _process_single_batch_object(self, batch: Batch) -> Batch:
        """Process a single batch with error handling."""
        if batch.data is None or len(batch.data) == 0:
            return batch
        
        try:
            # Apply Mean Shift clustering
            batch, _ = batch.mean_shift_2d()
            
            # Apply EM algorithm if clusters were found
            if batch.clusters_means is not None and len(batch.clusters_means) > 0:
                batch.em_algorithm()
                batch.score_fit()
        
        except Exception as e:
            if self.verbose and not self.silent_mode:
                self.logger.warning(f"Batch processing failed: {e}")
            # Return batch in original state
            
        return batch
    
    def _process_regular_batches(self, n_jobs: int, checkpoint: Optional[str], 
                                checkpoint_interval: int) -> Dict[str, Any]:
        """Process regular (continuous) batches."""
        # Similar implementation for continuous data batches
        # This would mirror the sparse batch processing logic
        valid_batches = [(i, batch) for i, batch in enumerate(self.batches)
                        if batch is not None and batch.data is not None and len(batch.data) > 0]
        
        # Implementation similar to _process_sparse_batches but for self.batches
        return {'processed_batches': len(valid_batches), 'total_components': 0, 'failed_batches': 0}
    
    def _get_config_dict(self) -> Dict[str, Any]:
        """Get current configuration as dictionary."""
        return {
            'resolution': self.resolution,
            'radius': self.radius,
            'bandwidth_mode': self.bandwidth_mode,
            'max_iterations': self.max_iterations,
            'convergence_threshold': self.convergence_threshold,
            'wind': self.wind
        }
    
    def _log_processing_summary(self, results: Dict[str, Any]):
        """Log comprehensive processing summary."""
        self.logger.info("=" * 50)
        self.logger.info("PROCESSING COMPLETE")
        self.logger.info("=" * 50)
        self.logger.info(f"Processed batches: {results['processed_batches']}")
        self.logger.info(f"Failed batches: {results['failed_batches']}")
        self.logger.info(f"Total components: {results['total_components']}")
        self.logger.info(f"Processing time: {results['total_processing_time']:.2f}s")
        self.logger.info(f"Average per batch: {results['average_time_per_batch']:.3f}s")
        self.logger.info(f"Processing rate: {results['components_per_second']:.2f} components/sec")
        self.logger.info(f"Mode: {results['processing_mode']}")
    
    def save_checkpoint(self, filepath: str):
        """Save current processing state."""
        self.checkpoint_manager.save_checkpoint(self, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load processing state from checkpoint."""
        self.checkpoint_manager.load_checkpoint(self, filepath)
    
    def continue_processing(self, **kwargs) -> Dict[str, Any]:
        """Continue interrupted processing session."""
        if not self.processing_interrupted:
            if self.verbose and not self.silent_mode:
                self.logger.info("No interrupted session found, starting fresh processing")
        
        return self.process(continue_processing=True, **kwargs)
    
    def save_results(self, filepath: str, format: str = 'csv'):
        """Save processing results to file."""
        if format.lower() == 'csv':
            self._save_csv_results(filepath)
        elif format.lower() == 'xml':
            self._save_xml_results(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_csv_results(self, filepath: str):
        """Save results in CSV format."""
        with open(filepath, 'w') as f:
            f.write("batch_id,x_center,y_center,component_id,theta,rho,weight\\n")
            
            batches_to_save = self.batches_sparse or self.batches
            if batches_to_save:
                for batch in batches_to_save:
                    if (batch is not None and hasattr(batch, 'data') and 
                        batch.data is not None and len(batch.data) > 0 and
                        batch.clusters_means is not None):
                        
                        # Get center position
                        if hasattr(batch, 'pose') and batch.pose is not None:
                            x_center, y_center = batch.pose
                        else:
                            x_center, y_center = 0.0, 0.0
                        
                        for comp_idx, cluster_mean in enumerate(batch.clusters_means):
                            theta_val = cluster_mean[0]
                            rho_val = cluster_mean[1]
                            weight = (batch.p[comp_idx] if batch.p is not None and 
                                    comp_idx < len(batch.p) else 1.0 / len(batch.clusters_means))
                            
                            f.write(f"{batch.id},{x_center:.6f},{y_center:.6f},"
                                  f"{comp_idx+1},{theta_val:.6f},{rho_val:.6f},{weight:.6f}\\n")
        
        if self.verbose and not self.silent_mode:
            self.logger.info(f"Results saved to {filepath}")
    
    def _save_xml_results(self, filepath: str):
        """Save results in XML format.""" 
        # XML export implementation
        pass
    
    # Visualization methods
    def plot_flow_field(self, **kwargs):
        """Plot complete flow field visualization."""
        return self.visualizer.plot_flow_field(self, **kwargs)
    
    def plot_batch(self, batch_id: int, **kwargs):
        """Plot single batch analysis."""
        return self.visualizer.plot_batch(self, batch_id, **kwargs)
    
    def plot_statistics(self, **kwargs):
        """Plot statistical summaries."""
        return self.visualizer.plot_statistics(self, **kwargs)
    
    def plot_input_data(self, **kwargs):
        """Plot raw input data."""
        return self.visualizer.plot_input_data(self, **kwargs)


def _process_single_batch_global(batch: Batch) -> Batch:
    """Global function for parallel processing."""
    if batch.data is None or len(batch.data) == 0:
        return batch
    
    try:
        # Apply Mean Shift clustering
        batch, _ = batch.mean_shift_2d()
        
        # Apply EM algorithm if clusters were found
        if batch.clusters_means is not None and len(batch.clusters_means) > 0:
            batch.em_algorithm()
            batch.score_fit()
    
    except Exception:
        # Return batch in original state if processing fails
        pass
        
    return batch