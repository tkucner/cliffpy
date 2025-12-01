#!/usr/bin/env python3
"""
CLiFF-map Package Comparison Demo

This script demonstrates both the standard and experimental high-performance 
CLiFF-map implementations, showing the performance differences.
"""

import sys
import os
import numpy as np
import time

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("CLiFF-map Package Comparison Demo")
print("="*50)

# Test both packages
def generate_test_data(n_points=3000):
    """Generate test data for comparison."""
    np.random.seed(42)
    
    x = np.random.uniform(-8, 8, n_points)
    y = np.random.uniform(-5, 5, n_points)
    
    # Create multiple flow patterns
    directions = np.zeros(n_points)
    speeds = np.zeros(n_points)
    
    for i in range(n_points):
        xi, yi = x[i], y[i]
        
        if xi > 2:  # Right region: upward flow
            directions[i] = np.pi/2 + 0.2 * np.random.normal()
            speeds[i] = 1.5 + 0.3 * np.random.normal()
        elif xi < -2:  # Left region: downward flow  
            directions[i] = -np.pi/2 + 0.2 * np.random.normal()
            speeds[i] = 1.3 + 0.3 * np.random.normal()
        else:  # Central region: mixed flow
            directions[i] = np.random.uniform(0, 2*np.pi)
            speeds[i] = 1.0 + 0.4 * np.random.normal()
    
    # Ensure valid values
    directions = np.mod(directions, 2*np.pi)
    speeds = np.maximum(0.1, speeds)
    
    return np.column_stack([x, y, directions, speeds])

# Generate test data
data = generate_test_data(3000)
print(f"Generated test data: {data.shape[0]} points")

# Test experimental package
print("\n1. EXPERIMENTAL HIGH-PERFORMANCE PACKAGE")
print("-" * 40)

try:
    from cliffmap_experimental import DynamicMapFast
    
    cliff_fast = DynamicMapFast(
        batch_size=100,
        use_gpu=False,
        adaptive_batching=True,
        spatial_index='grid',
        parallel=False,  # For fair comparison
        verbose=False
    )
    
    start_time = time.time()
    cliff_fast.fit(data)
    fast_time = time.time() - start_time
    
    print(f"✓ Processing time: {fast_time:.3f} seconds")
    print(f"✓ Components found: {len(cliff_fast.components)}")
    print(f"✓ Processing rate: {len(data)/fast_time:.0f} points/second")
    
    # Get performance breakdown
    perf_report = cliff_fast.get_performance_report()
    print(f"  Performance breakdown:")
    for key, value in perf_report['timing'].items():
        if value > 0:
            print(f"    {key}: {value:.3f}s")
    
    # Save results
    cliff_fast.save_xml('comparison_fast_results.xml')
    cliff_fast.save_csv('comparison_fast_results.csv')
    print(f"✓ Results saved to comparison_fast_results.*")
    
except Exception as e:
    print(f"✗ Experimental package failed: {e}")
    cliff_fast = None
    fast_time = None

# Test standard package  
print("\n2. STANDARD PACKAGE")
print("-" * 40)

try:
    from cliffmap import DynamicMap
    
    cliff_standard = DynamicMap(
        batch_size=50,  # Smaller batch for compatibility
        parallel=False,
        verbose=False
    )
    
    start_time = time.time()
    cliff_standard.fit(data)
    standard_time = time.time() - start_time
    
    print(f"✓ Processing time: {standard_time:.3f} seconds")
    print(f"✓ Components found: {len(cliff_standard.components)}")
    print(f"✓ Processing rate: {len(data)/standard_time:.0f} points/second")
    
    # Save results
    cliff_standard.save_xml('comparison_standard_results.xml')
    cliff_standard.save_csv('comparison_standard_results.csv')
    print(f"✓ Results saved to comparison_standard_results.*")
    
except Exception as e:
    print(f"✗ Standard package failed: {e}")
    cliff_standard = None
    standard_time = None

# Compare results
print("\n3. PERFORMANCE COMPARISON")
print("-" * 40)

if fast_time and standard_time:
    speedup = standard_time / fast_time
    print(f"Speedup achieved: {speedup:.2f}x")
    print(f"Time saved: {standard_time - fast_time:.3f} seconds")
    print(f"Efficiency gain: {(1 - fast_time/standard_time)*100:.1f}%")
    
    # Compare accuracy (component counts)
    if cliff_fast and cliff_standard:
        fast_components = len(cliff_fast.components)
        standard_components = len(cliff_standard.components)
        
        print(f"\nAccuracy comparison:")
        print(f"  Experimental components: {fast_components}")
        print(f"  Standard components: {standard_components}")
        
        if abs(fast_components - standard_components) <= 2:
            print(f"  ✓ Component counts similar (within acceptable range)")
        else:
            print(f"  ⚠ Component counts differ significantly")

elif fast_time:
    print(f"Only experimental package completed successfully")
    print(f"Processing rate: {len(data)/fast_time:.0f} points/second")
elif standard_time:
    print(f"Only standard package completed successfully")
    print(f"Processing rate: {len(data)/standard_time:.0f} points/second")
else:
    print("Neither package completed successfully")

# Feature comparison
print("\n4. FEATURE COMPARISON")
print("-" * 40)

print("Standard Package Features:")
print("  ✓ Mature, well-tested codebase")
print("  ✓ Full feature set")
print("  ✓ Extensive validation")
print("  ✓ Production-ready")

print("\nExperimental Package Features:")
print("  ✓ Vectorized operations")
print("  ✓ Adaptive batch sizing")
print("  ✓ Spatial indexing")
print("  ✓ GPU acceleration support")
print("  ✓ JIT compilation ready")
print("  ⚠ Experimental status")

print("\n5. RECOMMENDATIONS")
print("-" * 40)

print("Use Standard Package when:")
print("  • Production applications")
print("  • Maximum reliability needed")
print("  • Small to medium datasets (<10K points)")
print("  • All features required")

print("\nUse Experimental Package when:")
print("  • Performance is critical")
print("  • Large datasets (>10K points)")
print("  • Research and prototyping")
print("  • GPU resources available")

# Cleanup
print(f"\n6. CLEANUP")
print("-" * 40)

cleanup_files = [
    'comparison_fast_results.xml',
    'comparison_fast_results.csv',
    'comparison_standard_results.xml',
    'comparison_standard_results.csv'
]

for filename in cleanup_files:
    try:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"✓ Removed {filename}")
    except:
        pass

print("\n" + "="*50)
print("COMPARISON DEMO COMPLETED")
print("="*50)

if fast_time and standard_time:
    print(f"🚀 Experimental package is {standard_time/fast_time:.2f}x faster!")
print("📦 Both packages provide CLiFF-map analysis")
print("🎯 Choose based on your specific needs")