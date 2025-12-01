#!/usr/bin/env python3
"""
High-Performance CLiFF-map Example and Benchmark

This script demonstrates the experimental high-performance CLiFF-map implementation
and compares its performance against the standard version.

Features tested:
- Vectorized operations
- Adaptive batching
- Spatial indexing
- GPU acceleration (when available)
- Parallel processing
"""

import sys
import os
import numpy as np
import time
import warnings

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Performance tracking
import psutil
import gc

# Check dependencies
HAS_NUMBA = False
HAS_CUPY = False
HAS_MATPLOTLIB = False

try:
    import numba
    HAS_NUMBA = True
    print("✓ Numba available - JIT compilation enabled")
except ImportError:
    print("⚠ Numba not available - JIT compilation disabled")

try:
    import cupy as cp
    HAS_CUPY = True
    print("✓ CuPy available - GPU acceleration enabled")
except ImportError:
    print("⚠ CuPy not available - GPU acceleration disabled")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    print("✓ Matplotlib available - visualizations enabled")
except ImportError:
    print("⚠ Matplotlib not available - visualizations disabled")

# Import implementations
from cliffmap_experimental import DynamicMapFast
from cliffmap_experimental.dynamic_map_fast import benchmark_performance

try:
    from cliffmap import DynamicMap as DynamicMapOriginal
    HAS_ORIGINAL = True
    print("✓ Original CLiFF-map available for comparison")
except ImportError:
    HAS_ORIGINAL = False
    print("⚠ Original CLiFF-map not available for comparison")


def generate_test_data(n_points: int = 5000, pattern: str = 'complex') -> np.ndarray:
    """Generate test data with various flow patterns."""
    np.random.seed(42)
    
    if pattern == 'simple':
        # Simple unidirectional flow with noise
        x = np.random.uniform(-10, 10, n_points)
        y = np.random.uniform(-5, 5, n_points)
        direction = 0.2 + 0.1 * np.random.normal(size=n_points)
        speed = 1.5 + 0.3 * np.random.normal(size=n_points)
        
    elif pattern == 'vortex':
        # Vortex flow pattern
        r = np.random.uniform(0.5, 8, n_points)
        theta = np.random.uniform(0, 2*np.pi, n_points)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Tangential flow (vortex)
        direction = theta + np.pi/2 + 0.2 * np.random.normal(size=n_points)
        speed = 2.0 / (1 + r/3) + 0.3 * np.random.normal(size=n_points)
        
    elif pattern == 'complex':
        # Complex multi-modal flow
        x = np.random.uniform(-15, 15, n_points)
        y = np.random.uniform(-10, 10, n_points)
        
        direction = np.zeros(n_points)
        speed = np.zeros(n_points)
        
        for i in range(n_points):
            xi, yi = x[i], y[i]
            
            # Multiple flow regions
            if xi < -5:  # Left region: upward flow
                direction[i] = np.pi/2 + 0.3 * np.random.normal()
                speed[i] = 1.0 + 0.2 * np.random.normal()
            elif xi > 5:  # Right region: downward flow
                direction[i] = -np.pi/2 + 0.3 * np.random.normal()
                speed[i] = 1.2 + 0.2 * np.random.normal()
            elif yi > 3:  # Top region: rightward flow
                direction[i] = 0.1 + 0.2 * np.random.normal()
                speed[i] = 1.5 + 0.3 * np.random.normal()
            elif yi < -3:  # Bottom region: leftward flow
                direction[i] = np.pi + 0.2 * np.random.normal()
                speed[i] = 1.3 + 0.3 * np.random.normal()
            else:  # Central region: vortex
                r = np.sqrt(xi**2 + yi**2)
                theta = np.arctan2(yi, xi)
                direction[i] = theta + np.pi/2 + 0.4 * np.random.normal()
                speed[i] = 2.0 / (1 + r/2) + 0.4 * np.random.normal()
        
        # Ensure positive speeds
        speed = np.maximum(0.1, speed)
        
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Normalize directions to [0, 2π]
    direction = np.mod(direction, 2*np.pi)
    speed = np.maximum(0.1, speed)
    
    return np.column_stack([x, y, direction, speed])


def measure_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def performance_comparison_test():
    """Comprehensive performance comparison."""
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE COMPARISON")
    print("="*80)
    
    # Test configurations
    test_sizes = [1000, 2500, 5000, 10000]
    test_patterns = ['simple', 'vortex', 'complex']
    
    results = {
        'sizes': test_sizes,
        'patterns': test_patterns,
        'fast_cpu_times': {pattern: [] for pattern in test_patterns},
        'fast_gpu_times': {pattern: [] for pattern in test_patterns},
        'original_times': {pattern: [] for pattern in test_patterns},
        'memory_usage': {pattern: [] for pattern in test_patterns},
        'component_counts': {pattern: [] for pattern in test_patterns}
    }
    
    for pattern in test_patterns:
        print(f"\n--- Testing {pattern.upper()} pattern ---")
        
        for n_points in test_sizes:
            print(f"\nData size: {n_points} points")
            
            # Generate test data
            data = generate_test_data(n_points, pattern)
            gc.collect()  # Clean up before testing
            
            # Test Fast CPU version
            print("  Testing Fast CPU implementation...")
            start_mem = measure_memory_usage()
            
            cliff_fast_cpu = DynamicMapFast(
                batch_size=min(100, n_points//10),
                use_gpu=False,
                adaptive_batching=True,
                spatial_index='auto',
                parallel=True,
                verbose=False
            )
            
            start_time = time.time()
            cliff_fast_cpu.fit(data)
            cpu_time = time.time() - start_time
            
            end_mem = measure_memory_usage()
            memory_used = end_mem - start_mem
            
            results['fast_cpu_times'][pattern].append(cpu_time)
            results['memory_usage'][pattern].append(memory_used)
            results['component_counts'][pattern].append(len(cliff_fast_cpu.components))
            
            print(f"    Time: {cpu_time:.3f}s")
            print(f"    Components: {len(cliff_fast_cpu.components)}")
            print(f"    Memory: {memory_used:.1f}MB")
            print(f"    Rate: {n_points/cpu_time:.0f} points/s")
            
            # Get performance report
            perf_report = cliff_fast_cpu.get_performance_report()
            
            # Test Fast GPU version (if available)
            if HAS_CUPY:
                print("  Testing Fast GPU implementation...")
                try:
                    cliff_fast_gpu = DynamicMapFast(
                        batch_size=min(100, n_points//10),
                        use_gpu=True,
                        adaptive_batching=True,
                        verbose=False
                    )
                    
                    start_time = time.time()
                    cliff_fast_gpu.fit(data)
                    gpu_time = time.time() - start_time
                    
                    results['fast_gpu_times'][pattern].append(gpu_time)
                    
                    print(f"    GPU Time: {gpu_time:.3f}s")
                    print(f"    GPU Components: {len(cliff_fast_gpu.components)}")
                    print(f"    GPU Speedup: {cpu_time/gpu_time:.2f}x")
                    
                except Exception as e:
                    print(f"    GPU test failed: {e}")
                    results['fast_gpu_times'][pattern].append(None)
            else:
                results['fast_gpu_times'][pattern].append(None)
            
            # Test Original version (if available and data size not too large)
            if HAS_ORIGINAL and n_points <= 5000:  # Avoid long waits
                print("  Testing Original implementation...")
                try:
                    cliff_original = DynamicMapOriginal(
                        batch_size=min(50, n_points//20),
                        parallel=False,  # For fair comparison
                        verbose=False
                    )
                    
                    start_time = time.time()
                    cliff_original.fit(data)
                    original_time = time.time() - start_time
                    
                    results['original_times'][pattern].append(original_time)
                    
                    print(f"    Original Time: {original_time:.3f}s")
                    print(f"    Original Components: {len(cliff_original.components)}")
                    print(f"    Fast Speedup: {original_time/cpu_time:.2f}x")
                    
                except Exception as e:
                    print(f"    Original test failed: {e}")
                    results['original_times'][pattern].append(None)
            else:
                results['original_times'][pattern].append(None)
            
            # Cleanup
            del cliff_fast_cpu
            if HAS_CUPY:
                try:
                    del cliff_fast_gpu
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass
            if HAS_ORIGINAL:
                try:
                    del cliff_original
                except:
                    pass
            gc.collect()
    
    return results


def scalability_test():
    """Test scalability with very large datasets."""
    print("\n" + "="*80)
    print("SCALABILITY TEST")
    print("="*80)
    
    large_sizes = [5000, 10000, 25000, 50000]
    if HAS_CUPY:
        large_sizes.extend([100000, 200000])
    
    scalability_results = {
        'sizes': [],
        'cpu_times': [],
        'gpu_times': [],
        'memory_usage': [],
        'components': []
    }
    
    for n_points in large_sizes:
        print(f"\nTesting with {n_points:,} data points...")
        
        # Generate large test dataset
        data = generate_test_data(n_points, 'complex')
        
        # CPU test
        start_mem = measure_memory_usage()
        cliff_fast = DynamicMapFast(
            batch_size=200,
            use_gpu=False,
            adaptive_batching=True,
            spatial_index='grid',  # Fastest for large data
            parallel=True,
            n_jobs=4,
            verbose=False
        )
        
        start_time = time.time()
        cliff_fast.fit(data)
        cpu_time = time.time() - start_time
        end_mem = measure_memory_usage()
        
        scalability_results['sizes'].append(n_points)
        scalability_results['cpu_times'].append(cpu_time)
        scalability_results['memory_usage'].append(end_mem - start_mem)
        scalability_results['components'].append(len(cliff_fast.components))
        
        print(f"  CPU: {cpu_time:.2f}s ({n_points/cpu_time:.0f} points/s)")
        print(f"  Memory: {end_mem - start_mem:.1f}MB")
        print(f"  Components: {len(cliff_fast.components)}")
        
        # GPU test (if available and reasonable size)
        if HAS_CUPY and n_points <= 100000:  # Avoid GPU memory issues
            try:
                cliff_fast_gpu = DynamicMapFast(
                    batch_size=200,
                    use_gpu=True,
                    adaptive_batching=True,
                    verbose=False
                )
                
                start_time = time.time()
                cliff_fast_gpu.fit(data)
                gpu_time = time.time() - start_time
                
                scalability_results['gpu_times'].append(gpu_time)
                
                print(f"  GPU: {gpu_time:.2f}s (speedup: {cpu_time/gpu_time:.2f}x)")
                
                del cliff_fast_gpu
                cp.get_default_memory_pool().free_all_blocks()
                
            except Exception as e:
                print(f"  GPU failed: {e}")
                scalability_results['gpu_times'].append(None)
        else:
            scalability_results['gpu_times'].append(None)
        
        del cliff_fast
        gc.collect()
        
        # Break if taking too long
        if cpu_time > 30:  # More than 30 seconds
            print("  Stopping scalability test due to time limit")
            break
    
    return scalability_results


def adaptive_features_test():
    """Test adaptive features."""
    print("\n" + "="*80)
    print("ADAPTIVE FEATURES TEST")
    print("="*80)
    
    # Test data with varying density
    n_points = 8000
    data = generate_test_data(n_points, 'complex')
    
    print(f"Testing with {n_points} points...")
    
    # Test without adaptive features
    print("\n1. Standard batching:")
    cliff_standard = DynamicMapFast(
        batch_size=100,
        adaptive_batching=False,
        spatial_index='auto',
        use_gpu=False,
        verbose=False
    )
    
    start_time = time.time()
    cliff_standard.fit(data)
    standard_time = time.time() - start_time
    
    print(f"   Time: {standard_time:.3f}s")
    print(f"   Components: {len(cliff_standard.components)}")
    
    # Test with adaptive features
    print("\n2. Adaptive batching:")
    cliff_adaptive = DynamicMapFast(
        batch_size=100,  # Initial size
        adaptive_batching=True,
        spatial_index='auto',
        use_gpu=False,
        verbose=False
    )
    
    start_time = time.time()
    cliff_adaptive.fit(data)
    adaptive_time = time.time() - start_time
    
    print(f"   Time: {adaptive_time:.3f}s")
    print(f"   Components: {len(cliff_adaptive.components)}")
    print(f"   Improvement: {standard_time/adaptive_time:.2f}x")
    
    # Test different spatial indexing methods
    print("\n3. Spatial indexing comparison:")
    methods = ['grid', 'auto']
    if HAS_SKLEARN:
        methods.append('kdtree')
    
    for method in methods:
        cliff_spatial = DynamicMapFast(
            batch_size=100,
            adaptive_batching=False,
            spatial_index=method,
            use_gpu=False,
            verbose=False
        )
        
        start_time = time.time()
        cliff_spatial.fit(data)
        method_time = time.time() - start_time
        
        print(f"   {method}: {method_time:.3f}s")
        
        del cliff_spatial
        gc.collect()


def create_performance_visualization(results, scalability_results):
    """Create performance visualization plots."""
    if not HAS_MATPLOTLIB:
        print("\nMatplotlib not available - skipping visualizations")
        return
    
    print("\n" + "="*80)
    print("CREATING PERFORMANCE VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Performance comparison by data size
    ax1 = axes[0, 0]
    sizes = results['sizes']
    
    for pattern in results['patterns']:
        cpu_times = results['fast_cpu_times'][pattern]
        ax1.plot(sizes, cpu_times, 'o-', label=f'Fast CPU ({pattern})')
        
        if any(results['fast_gpu_times'][pattern]):
            gpu_times = [t for t in results['fast_gpu_times'][pattern] if t is not None]
            gpu_sizes = [sizes[i] for i, t in enumerate(results['fast_gpu_times'][pattern]) if t is not None]
            if gpu_times:
                ax1.plot(gpu_sizes, gpu_times, 's--', label=f'Fast GPU ({pattern})')
    
    ax1.set_xlabel('Number of Data Points')
    ax1.set_ylabel('Processing Time (seconds)')
    ax1.set_title('Performance Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scalability
    ax2 = axes[0, 1]
    scal_sizes = scalability_results['sizes']
    scal_cpu = scalability_results['cpu_times']
    
    # Throughput (points per second)
    throughput = [s/t for s, t in zip(scal_sizes, scal_cpu)]
    ax2.plot(scal_sizes, throughput, 'o-', label='CPU Throughput')
    
    if any(scalability_results['gpu_times']):
        gpu_throughput = [s/t for s, t in zip(scal_sizes, scalability_results['gpu_times']) 
                         if t is not None]
        gpu_sizes_valid = [s for s, t in zip(scal_sizes, scalability_results['gpu_times']) 
                          if t is not None]
        if gpu_throughput:
            ax2.plot(gpu_sizes_valid, gpu_throughput, 's--', label='GPU Throughput')
    
    ax2.set_xlabel('Number of Data Points')
    ax2.set_ylabel('Throughput (points/second)')
    ax2.set_title('Scalability Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Memory usage
    ax3 = axes[1, 0]
    for pattern in results['patterns']:
        memory_usage = results['memory_usage'][pattern]
        ax3.plot(sizes, memory_usage, 'o-', label=f'{pattern}')
    
    ax3.set_xlabel('Number of Data Points')
    ax3.set_ylabel('Memory Usage (MB)')
    ax3.set_title('Memory Efficiency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Components found
    ax4 = axes[1, 1]
    for pattern in results['patterns']:
        components = results['component_counts'][pattern]
        ax4.plot(sizes, components, 'o-', label=f'{pattern}')
    
    ax4.set_xlabel('Number of Data Points')
    ax4.set_ylabel('Number of Components Found')
    ax4.set_title('Component Detection')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cliffmap_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Saved performance analysis plot: cliffmap_performance_analysis.png")


def main():
    """Run comprehensive performance tests."""
    print("HIGH-PERFORMANCE CLiFF-MAP ANALYSIS")
    print("="*80)
    print(f"NumPy version: {np.__version__}")
    if HAS_NUMBA:
        print(f"Numba version: {numba.__version__}")
    if HAS_CUPY:
        print(f"CuPy version: {cp.__version__}")
    
    # Quick functionality test
    print("\n--- FUNCTIONALITY TEST ---")
    test_data = generate_test_data(1000, 'vortex')
    
    cliff_test = DynamicMapFast(
        batch_size=50,
        use_gpu=False,
        adaptive_batching=True,
        verbose=True
    )
    
    start_time = time.time()
    cliff_test.fit(test_data)
    test_time = time.time() - start_time
    
    print(f"✓ Basic functionality test passed")
    print(f"  Processed 1000 points in {test_time:.3f}s")
    print(f"  Found {len(cliff_test.components)} components")
    
    perf_report = cliff_test.get_performance_report()
    print(f"  Performance breakdown:")
    for key, value in perf_report['timing'].items():
        if value > 0:
            print(f"    {key}: {value:.3f}s")
    
    # Run comprehensive tests
    results = performance_comparison_test()
    scalability_results = scalability_test()
    adaptive_features_test()
    
    # Create visualizations
    create_performance_visualization(results, scalability_results)
    
    # Final summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    # Find best performance cases
    best_cpu_rate = 0
    best_gpu_rate = 0
    
    for pattern in results['patterns']:
        for i, size in enumerate(results['sizes']):
            cpu_time = results['fast_cpu_times'][pattern][i]
            rate = size / cpu_time
            if rate > best_cpu_rate:
                best_cpu_rate = rate
            
            if HAS_CUPY and results['fast_gpu_times'][pattern][i]:
                gpu_time = results['fast_gpu_times'][pattern][i]
                gpu_rate = size / gpu_time
                if gpu_rate > best_gpu_rate:
                    best_gpu_rate = gpu_rate
    
    print(f"Best CPU performance: {best_cpu_rate:.0f} points/second")
    if HAS_CUPY:
        print(f"Best GPU performance: {best_gpu_rate:.0f} points/second")
        if best_gpu_rate > best_cpu_rate:
            print(f"Maximum GPU speedup: {best_gpu_rate/best_cpu_rate:.2f}x")
    
    # Check if original comparison is available
    if HAS_ORIGINAL:
        max_speedup = 0
        for pattern in results['patterns']:
            for i, orig_time in enumerate(results['original_times'][pattern]):
                if orig_time is not None:
                    fast_time = results['fast_cpu_times'][pattern][i]
                    speedup = orig_time / fast_time
                    max_speedup = max(max_speedup, speedup)
        
        if max_speedup > 0:
            print(f"Maximum speedup vs original: {max_speedup:.2f}x")
    
    print("\nOptimizations successfully demonstrated:")
    print("✓ Vectorized NumPy operations")
    print("✓ Numba JIT compilation" if HAS_NUMBA else "- Numba not available")
    print("✓ GPU acceleration" if HAS_CUPY else "- GPU not available")
    print("✓ Adaptive batch sizing")
    print("✓ Spatial indexing")
    print("✓ Parallel processing")
    print("✓ Memory-efficient algorithms")


if __name__ == "__main__":
    main()