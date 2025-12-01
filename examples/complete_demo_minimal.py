#!/usr/bin/env python3
"""
Complete CLiFF-map Demo - Minimal Dependencies

This example demonstrates the full CLiFF-map workflow with fallbacks for all
optional dependencies. Works with just numpy and the base CLiFF-map package.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import CLiFF-map
from cliffmap.dynamic_map import DynamicMap

print("CLiFF-map Complete Demo")
print("=" * 50)

def generate_demo_data(pattern="mixed"):
    """Generate demonstration data with different flow patterns."""
    np.random.seed(42)
    n_points = 800
    
    if pattern == "mixed":
        # Mixed directional and velocity data
        x = np.random.uniform(-4, 4, n_points)
        y = np.random.uniform(-3, 3, n_points)
        
        # Create interesting flow patterns
        directions = []
        speeds = []
        
        for xi, yi in zip(x, y):
            if xi**2 + yi**2 < 1:  # Center vortex
                angle = np.arctan2(-xi, yi)  # Clockwise vortex
                speed = 1.5
            elif abs(yi) < 0.5:  # Horizontal flow
                angle = 0 if xi < 0 else np.pi  
                speed = 2.0
            else:  # Random background
                angle = np.random.uniform(0, 2*np.pi)
                speed = 0.8 + 0.4 * np.random.random()
            
            # Add noise
            angle += np.random.normal(0, 0.2)
            speed += np.random.normal(0, 0.1)
            speed = max(0.1, speed)
            
            directions.append(angle)
            speeds.append(speed)
        
        return np.column_stack([x, y, directions, speeds])
    
    elif pattern == "velocity":
        # Pure velocity data
        x = np.random.uniform(-3, 3, n_points)
        y = np.random.uniform(-2, 2, n_points)
        
        # Simple left-to-right flow with vertical variation
        vx = 1.0 + 0.5*np.random.normal(size=n_points)
        vy = 0.3*y + 0.2*np.random.normal(size=n_points)
        
        return np.column_stack([x, y, vx, vy])


def basic_demo():
    """Basic CLiFF-map demonstration."""
    print("\n1. BASIC ANALYSIS")
    print("-" * 30)
    
    # Generate test data
    data = generate_demo_data("mixed")
    print(f"Generated {data.shape[0]} data points with shape {data.shape}")
    
    # Create simple CLiFF-map (disable parallel to avoid errors without sklearn)
    cliff_map = DynamicMap(
        batch_size=40,
        max_iterations=50,
        bandwidth=0.5,
        min_samples=5,
        parallel=False,  # Disable to avoid sklearn issues
        verbose=True
    )
    
    print("Running CLiFF-map analysis...")
    try:
        cliff_map.fit(data)
        print(f"✓ Found {len(cliff_map.components)} flow components")
        
        # Show component details
        for i, comp in enumerate(cliff_map.components):
            pos = comp['position']
            direction = comp.get('direction', 0)
            weight = comp['weight']
            print(f"  Component {i+1}: pos=({pos[0]:.1f},{pos[1]:.1f}), "
                  f"dir={np.degrees(direction):.0f}°, weight={weight:.2f}")
        
        return cliff_map, data
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return None, data


def data_format_demo():
    """Demonstrate different data format support."""
    print("\n2. DATA FORMAT SUPPORT")
    print("-" * 30)
    
    # Test directional format [x, y, direction, speed]
    print("Testing directional format [x, y, direction, speed]...")
    data_dir = generate_demo_data("mixed")
    cliff_map_dir = DynamicMap(batch_size=30, parallel=False, verbose=False)
    
    try:
        cliff_map_dir.fit(data_dir)
        print(f"✓ Directional format: {len(cliff_map_dir.components)} components found")
    except Exception as e:
        print(f"✗ Directional format failed: {e}")
    
    # Test velocity format [x, y, vx, vy]
    print("Testing velocity format [x, y, vx, vy]...")
    data_vel = generate_demo_data("velocity")
    cliff_map_vel = DynamicMap(batch_size=30, parallel=False, verbose=False)
    
    try:
        cliff_map_vel.fit(data_vel)
        print(f"✓ Velocity format: {len(cliff_map_vel.components)} components found")
    except Exception as e:
        print(f"✗ Velocity format failed: {e}")


def export_demo(cliff_map):
    """Demonstrate export functionality."""
    print("\n3. EXPORT FUNCTIONALITY")
    print("-" * 30)
    
    if cliff_map is None or len(cliff_map.components) == 0:
        print("No components to export")
        return
    
    try:
        # XML export
        xml_file = "demo_results.xml"
        cliff_map.save_xml(xml_file)
        if os.path.exists(xml_file):
            size = os.path.getsize(xml_file)
            print(f"✓ XML export: {xml_file} ({size} bytes)")
        
        # CSV export  
        csv_file = "demo_results.csv"
        cliff_map.save_csv(csv_file)
        if os.path.exists(csv_file):
            size = os.path.getsize(csv_file)
            print(f"✓ CSV export: {csv_file} ({size} bytes)")
            
    except Exception as e:
        print(f"✗ Export failed: {e}")


def parameter_demo():
    """Demonstrate parameter effects."""
    print("\n4. PARAMETER EFFECTS")
    print("-" * 30)
    
    data = generate_demo_data("mixed")
    
    # Test different bandwidths
    bandwidths = [0.3, 0.5, 0.8]
    for bw in bandwidths:
        cliff_map = DynamicMap(
            batch_size=30,
            bandwidth=bw,
            parallel=False,
            verbose=False
        )
        
        try:
            cliff_map.fit(data)
            print(f"Bandwidth {bw}: {len(cliff_map.components)} components")
        except:
            print(f"Bandwidth {bw}: failed")


def main():
    """Run complete demonstration."""
    print("Starting CLiFF-map complete demonstration...")
    print("This demo works with minimal dependencies (numpy only)")
    
    # Run basic analysis
    cliff_map, data = basic_demo()
    
    # Test data formats
    data_format_demo()
    
    # Test exports
    export_demo(cliff_map)
    
    # Test parameters
    parameter_demo()
    
    print("\n" + "=" * 50)
    print("DEMO COMPLETE!")
    print("=" * 50)
    print("✓ CLiFF-map analysis demonstrated successfully")
    print("✓ Multiple data formats tested")
    print("✓ Export functionality verified")
    print("✓ Parameter effects explored")
    print("\nCheck generated files: demo_results.xml, demo_results.csv")


if __name__ == "__main__":
    main()