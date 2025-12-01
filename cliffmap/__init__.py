"""
CLiFF-map: Circular-Linear Flow Field mapping for dynamic environments

A comprehensive Python implementation of CLiFF-map (Circular-Linear Flow Field mapping)
for analyzing dynamic environments with flow patterns. This package provides enhanced
features including parallel processing, progress monitoring, checkpointing, and 
advanced visualization capabilities.

Key Features:
- Circular-Linear Flow Field mapping using Mean Shift and EM algorithms
- Parallel processing support with ThreadPoolExecutor
- Progress monitoring with tqdm progress bars
- Comprehensive checkpointing and state management
- Advanced visualization tools with matplotlib and seaborn
- Support for multiple data formats (CSV, XML export)
- Robust error handling and logging
- Memory-efficient batch processing
- Detailed training history and convergence monitoring

Classes:
- DynamicMap: Main flow field mapping class with enhanced features
- Batch: Optimized batch processing for flow data
- FlowFieldVisualizer: Comprehensive visualization toolkit
- CheckpointManager: Advanced checkpointing and state management

Usage:
    >>> from cliffmap import DynamicMap, FlowFieldVisualizer
    >>> 
    >>> # Load your flow data (x, y, direction, speed)
    >>> data = load_your_data()
    >>> 
    >>> # Initialize with enhanced features
    >>> cliff_map = DynamicMap(
    ...     batch_size=100,
    ...     parallel=True,
    ...     progress=True,
    ...     verbose=True
    ... )
    >>> 
    >>> # Fit the model
    >>> cliff_map.fit(data)
    >>> 
    >>> # Visualize results
    >>> visualizer = FlowFieldVisualizer()
    >>> visualizer.plot_components(cliff_map, save_path="results.png")

Original MATLAB implementation by:
- Zhi Yan
- Tom Duckett  
- Nicola Bellotto

Python implementation with enhancements by the CLiFF-map development team.
"""

__version__ = "1.0.0"
__author__ = "CLiFF-map Development Team"
__email__ = "support@cliffmap.org"
__license__ = "MIT"
__description__ = "Circular-Linear Flow Field mapping for dynamic environment analysis"

# Core imports
from .dynamic_map import DynamicMap
from .batch import Batch

# Utilities
try:
    from .utils import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import all utilities: {e}")
    # Provide fallback for essential functions
    import numpy as np
    
    def WangDivergence(p, q):
        """Fallback Wang divergence implementation."""
        return np.sum(np.abs(p - q))
    
    def circular_mean(angles, weights=None):
        """Fallback circular mean implementation."""
        if weights is None:
            weights = np.ones(len(angles))
        return np.angle(np.sum(weights * np.exp(1j * angles)) / np.sum(weights))

# Enhanced visualization
try:
    from .visualization import FlowFieldVisualizer, plot_data_distribution
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import visualization module: {e}")
    FlowFieldVisualizer = None
    plot_data_distribution = None

# Checkpointing and state management
try:
    from .checkpoint import CheckpointManager, create_training_session, auto_checkpoint, resume_training
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import checkpoint module: {e}")
    CheckpointManager = None
    create_training_session = None
    auto_checkpoint = None
    resume_training = None

# Main exports
__all__ = [
    # Core classes
    'DynamicMap',
    'Batch',
    
    # Visualization
    'FlowFieldVisualizer',
    'plot_data_distribution',
    
    # Checkpointing
    'CheckpointManager',
    'create_training_session',
    'auto_checkpoint',
    'resume_training',
    
    # Utilities (from utils.py)
    'cart2pol',
    'pol2cart',
    'wrap_to_2pi',
    'circular_mean',
    'circular_variance',
    'WangDivergence',
    
    # Metadata
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    '__description__'
]

# Package configuration
DEFAULT_CONFIG = {
    'batch_size': 100,
    'max_iterations': 200,
    'bandwidth': 0.5,
    'min_samples': 10,
    'convergence_threshold': 1e-4,
    'parallel': True,
    'n_jobs': 2,
    'verbose': False,
    'progress': True
}

def get_version_info():
    """Get detailed version information."""
    import sys
    import platform
    
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = "Not installed"
    
    try:
        import scipy
        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = "Not installed"
    
    try:
        import matplotlib
        matplotlib_version = matplotlib.__version__
    except ImportError:
        matplotlib_version = "Not installed"
    
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = "Not installed"
    
    return {
        'cliffmap_version': __version__,
        'python_version': sys.version,
        'platform': platform.platform(),
        'dependencies': {
            'numpy': numpy_version,
            'scipy': scipy_version,
            'matplotlib': matplotlib_version,
            'scikit-learn': sklearn_version
        }
    }

def print_version_info():
    """Print detailed version and dependency information."""
    info = get_version_info()
    
    print(f"CLiFF-map version: {info['cliffmap_version']}")
    print(f"Python version: {info['python_version']}")
    print(f"Platform: {info['platform']}")
    print(f"Dependencies:")
    
    for package, version in info['dependencies'].items():
        status = "✓" if version != "Not installed" else "✗"
        print(f"  {status} {package}: {version}")