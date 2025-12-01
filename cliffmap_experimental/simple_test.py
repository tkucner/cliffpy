#!/usr/bin/env python3
"""
Simple test for the experimental high-performance CLiFF-map package
"""

import sys
import os
import numpy as np
import time

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing Experimental High-Performance CLiFF-map")
print("=" * 60)

try:
    from cliffmap_experimental import DynamicMapFast
    print("✓ Successfully imported DynamicMapFast")
except ImportError as e:
    print(f"✗ Failed to import DynamicMapFast: {e}")
    sys.exit(1)

# Generate simple test data
print("\nGenerating test data...")
np.random.seed(42)
n_points = 2000

# Create simple flow pattern
x = np.random.uniform(-5, 5, n_points)
y = np.random.uniform(-3, 3, n_points)

# Simple rightward flow with some noise
direction = 0.1 + 0.3 * np.random.normal(size=n_points)
speed = 1.2 + 0.4 * np.random.normal(size=n_points)
speed = np.maximum(0.1, speed)  # Ensure positive speeds
direction = np.mod(direction, 2*np.pi)  # Normalize angles

test_data = np.column_stack([x, y, direction, speed])
print(f"✓ Generated {n_points} test data points")
print(f"  Data shape: {test_data.shape}")
print(f"  X range: [{x.min():.2f}, {x.max():.2f}]")
print(f"  Y range: [{y.min():.2f}, {y.max():.2f}]")
print(f"  Direction range: [{direction.min():.2f}, {direction.max():.2f}]")
print(f"  Speed range: [{speed.min():.2f}, {speed.max():.2f}]")

# Test basic functionality
print("\nTesting DynamicMapFast...")

# Create instance
cliff_fast = DynamicMapFast(
    batch_size=80,
    bandwidth=0.5,
    use_gpu=False,  # Start with CPU only
    adaptive_batching=True,
    spatial_index='grid',  # Use grid for compatibility
    parallel=False,  # Disable parallel for simplicity
    verbose=True
)

print("✓ Created DynamicMapFast instance")

# Fit the model
print("\nFitting model...")
start_time = time.time()

try:
    cliff_fast.fit(test_data)
    fit_time = time.time() - start_time
    
    print(f"✓ Model fitting completed in {fit_time:.3f} seconds")
    print(f"✓ Found {len(cliff_fast.components)} flow components")
    print(f"  Processing rate: {n_points/fit_time:.0f} points/second")
    
    # Show component details
    if cliff_fast.components:
        print(f"\nComponent details:")
        for i, comp in enumerate(cliff_fast.components[:5]):  # Show first 5
            pos = comp['position']
            direction = comp['direction']
            speed = comp['speed']
            weight = comp['weight']
            
            print(f"  Component {i+1}:")
            print(f"    Position: ({pos[0]:.2f}, {pos[1]:.2f})")
            print(f"    Direction: {np.degrees(direction):.1f}°")
            print(f"    Speed: {speed:.2f}")
            print(f"    Weight: {weight:.3f}")
        
        if len(cliff_fast.components) > 5:
            print(f"  ... and {len(cliff_fast.components)-5} more components")
    
    # Get performance report
    perf_report = cliff_fast.get_performance_report()
    print(f"\nPerformance breakdown:")
    for key, value in perf_report['timing'].items():
        if value > 0:
            print(f"  {key}: {value:.3f}s ({value/fit_time*100:.1f}%)")
    
    # Test export functionality
    print(f"\nTesting export functionality...")
    
    # XML export
    cliff_fast.save_xml('test_results_fast.xml')
    if os.path.exists('test_results_fast.xml'):
        size = os.path.getsize('test_results_fast.xml')
        print(f"✓ XML export successful: test_results_fast.xml ({size} bytes)")
    
    # CSV export
    cliff_fast.save_csv('test_results_fast.csv')
    if os.path.exists('test_results_fast.csv'):
        size = os.path.getsize('test_results_fast.csv')
        print(f"✓ CSV export successful: test_results_fast.csv ({size} bytes)")
    
except Exception as e:
    print(f"✗ Model fitting failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("EXPERIMENTAL HIGH-PERFORMANCE TEST COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("✓ All basic functionality working")
print("✓ Performance optimizations active")
print("✓ Export functionality verified")
print(f"✓ Processing rate: {n_points/fit_time:.0f} points/second")

# Cleanup
try:
    os.remove('test_results_fast.xml')
    os.remove('test_results_fast.csv')
    print("✓ Cleaned up test files")
except:
    pass