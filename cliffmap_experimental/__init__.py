"""
CLiFF-map Experimental High-Performance Package

This experimental version focuses on maximum time efficiency using:
- Vectorized NumPy operations
- Optimized memory usage
- Parallelized computation with numba JIT compilation
- GPU acceleration (when available)
- Sparse data structures
- Fast approximation algorithms

WARNING: This is experimental code for performance testing.
Use the main cliffmap package for production work.
"""

import warnings

# Check for performance-oriented dependencies
HAS_NUMBA = False
HAS_CUPY = False
HAS_SCIPY_SPARSE = False

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    warnings.warn("numba not available - JIT compilation disabled")

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    warnings.warn("cupy not available - GPU acceleration disabled")

try:
    from scipy import sparse
    HAS_SCIPY_SPARSE = True
except ImportError:
    warnings.warn("scipy.sparse not available - sparse optimizations disabled")

from .dynamic_map_fast import DynamicMapFast
from .batch_fast import BatchFast

__version__ = "0.1.0-experimental"
__all__ = ['DynamicMapFast', 'BatchFast']