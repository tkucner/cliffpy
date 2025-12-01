#!/usr/bin/env python3
"""
Air Flow Analysis Example using CLiFF-map

This example demonstrates how to analyze air flow patterns using the enhanced CLiFF-map
algorithm with automatic column detection and various data formats.

Dependencies (optional):
- pandas: for advanced data loading (fallback: synthetic data)
- matplotlib: for visualization (fallback: text output)
- scikit-learn: for additional processing (fallback: basic algorithms)
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for optional dependencies
HAS_PANDAS = False
HAS_MATPLOTLIB = False
HAS_SKLEARN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    print("Note: pandas not available, using synthetic data generation")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    print("Note: matplotlib not available, skipping visualizations")

try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    print("Note: scikit-learn not available, using basic algorithms")

# Import CLiFF-map
from cliffmap.dynamic_map import DynamicMap


def generate_air_flow_data(n_points=1000, random_seed=42):
    """Generate synthetic air flow data with realistic patterns."""
    np.random.seed(random_seed)
    
    # Create spatial domain
    x_range = (-5, 5)
    y_range = (-3, 3)
    
    # Generate points
    x = np.random.uniform(x_range[0], x_range[1], n_points)
    y = np.random.uniform(y_range[0], y_range[1], n_points)
    
    # Create flow patterns (multiple vortices and straight flow)
    vortex1_center = (2, 1)
    vortex2_center = (-2, -1)
    
    # Calculate flow directions based on position
    directions = []
    speeds = []
    
    for xi, yi in zip(x, y):
        # Distance to vortex centers
        d1 = np.sqrt((xi - vortex1_center[0])**2 + (yi - vortex1_center[1])**2)
        d2 = np.sqrt((xi - vortex2_center[0])**2 + (yi - vortex2_center[1])**2)
        
        if d1 < 1.5:  # Near vortex 1 (clockwise)
            angle = np.arctan2(yi - vortex1_center[1], xi - vortex1_center[0]) + np.pi/2
            speed = 2.0 * np.exp(-d1)
        elif d2 < 1.5:  # Near vortex 2 (counter-clockwise)
            angle = np.arctan2(yi - vortex2_center[1], xi - vortex2_center[0]) - np.pi/2
            speed = 2.0 * np.exp(-d2)
        else:  # Background flow (left to right)
            angle = 0.1 * np.sin(yi) + np.random.normal(0, 0.2)
            speed = 1.0 + 0.5 * np.random.normal()
        
        # Add noise
        angle += np.random.normal(0, 0.3)
        speed = max(0.1, speed + np.random.normal(0, 0.2))
        
        directions.append(angle)
        speeds.append(speed)
    
    # Create data array [x, y, direction, speed]
    data = np.column_stack([x, y, directions, speeds])
    
    print(f"Generated {n_points} synthetic air flow data points")
    print(f"Spatial domain: x∈{x_range}, y∈{y_range}")
    print(f"Speed range: {np.min(speeds):.2f} - {np.max(speeds):.2f}")
    print(f"Direction range: {np.degrees(np.min(directions)):.1f}° - {np.degrees(np.max(directions)):.1f}°")
    
    return data


def load_air_flow_data():
    """Load air flow data from file or generate synthetic data."""
    print("Loading air flow data...")
    
    # Try to load real data if available
    data_file = "data/air_flow.csv"
    if os.path.exists(data_file) and HAS_PANDAS:
        try:
            df = pd.read_csv(data_file)
            data = df.values
            print(f"Loaded real data from {data_file}: {data.shape}")
            return data
        except Exception as e:
            print(f"Failed to load {data_file}: {e}")
    
    # Generate synthetic data
    return generate_air_flow_data()


def basic_analysis_example():
    """Demonstrate basic CLiFF-map analysis."""
    print("🌪️  CLiFF-map Air Flow Analysis Example")
    print("=" * 60)
    print("BASIC ANALYSIS")
    print("=" * 60)
    
    # Load data
    data = load_air_flow_data()
    if data is None:
        print("❌ Failed to load data")
        return None, None
        
    print(f"\nData shape: {data.shape}")
    print(f"Columns: {data.shape[1]} (expected: x, y, direction, speed)")
    
    # Create CLiFF-map with reasonable parameters
    cliff_map = DynamicMap(
        batch_size=50,
        max_iterations=100,
        bandwidth=0.5,
        min_samples=5,
        parallel=True,
        n_jobs=2,
        verbose=True,
        progress=True
    )
    
    print("\nFitting CLiFF-map model...")
    try:
        cliff_map.fit(data)
        
        # Print results
        print(f"\nResults:")
        print(f"- Found {len(cliff_map.components)} flow components")
        print(f"- Data points: {data.shape[0]}")
        print(f"- Convergence: {'Yes' if hasattr(cliff_map, 'converged') and cliff_map.converged else 'Unknown'}")
        
        # Display component details
        print(f"\nComponent Details:")
        for i, component in enumerate(cliff_map.components):
            pos = component['position']
            direction_deg = np.degrees(component['direction']) % 360
            print(f"  Component {i+1}:")
            print(f"    Position: ({pos[0]:.2f}, {pos[1]:.2f})")
            print(f"    Direction: {direction_deg:.1f}°")
            print(f"    Weight: {component['weight']:.3f}")
    
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
    return cliff_map if 'cliff_map' in locals() else None, data


def visualization_example(cliff_map, data):
    """Demonstrate visualization capabilities."""
    print("\n" + "=" * 60)
    print("VISUALIZATION EXAMPLES")
    print("=" * 60)
    
    if cliff_map is None:
        print("No CLiFF-map object available for visualization")
        return
        
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available - skipping visualization")
        print("Install matplotlib with: pip install matplotlib")
        return
        
    try:
        # Basic plot of input data
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Raw data points
        plt.subplot(1, 3, 1)
        if data.shape[1] >= 4:
            scatter = plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], 
                                cmap='hsv', alpha=0.6, s=10)
            plt.colorbar(scatter, label='Direction (rad)')
        else:
            plt.scatter(data[:, 0], data[:, 1], alpha=0.6, s=10)
        plt.title('Input Data Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Components if available
        plt.subplot(1, 3, 2)
        plt.scatter(data[:, 0], data[:, 1], c='lightgray', alpha=0.3, s=5, label='Data points')
        
        if cliff_map.components:
            for i, comp in enumerate(cliff_map.components):
                pos = comp['position']
                direction = comp['direction']
                weight = comp['weight']
                
                # Plot component position
                plt.scatter(pos[0], pos[1], s=100*weight, c='red', 
                           alpha=0.7, edgecolors='black')
                
                # Plot direction arrow
                arrow_length = 0.5 * weight
                dx = arrow_length * np.cos(direction)
                dy = arrow_length * np.sin(direction)
                plt.arrow(pos[0], pos[1], dx, dy, head_width=0.1, 
                         head_length=0.1, fc='blue', ec='blue')
        
        plt.title(f'Flow Components ({len(cliff_map.components)})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 3: Data statistics
        plt.subplot(1, 3, 3)
        if data.shape[1] >= 4:
            directions = data[:, 2]
            speeds = data[:, 3]
            
            # Direction histogram
            plt.hist(directions, bins=20, alpha=0.7, label='Direction (rad)')
            plt.xlabel('Direction (radians)')
            plt.ylabel('Frequency')
            plt.title('Direction Distribution')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('air_flow_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved visualization as 'air_flow_analysis.png'")
        
    except Exception as e:
        print(f"Visualization failed: {e}")


def save_results_example(cliff_map, data):
    """Demonstrate saving results to different formats."""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    if cliff_map is None:
        print("No CLiFF-map object available for saving")
        return
    
    try:
        # Save to XML
        xml_filename = 'air_flow_results.xml'
        cliff_map.save_xml(xml_filename, include_metadata=True)
        print(f"✓ Saved XML results: {xml_filename}")
        
        # Save to CSV
        csv_filename = 'air_flow_components.csv'
        cliff_map.save_csv(csv_filename, include_velocity_components=True)
        print(f"✓ Saved CSV results: {csv_filename}")
        
        # Print file sizes
        for filename in [xml_filename, csv_filename]:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                print(f"  {filename}: {size} bytes")
        
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    """Run all examples."""
    print("🌪️  CLiFF-map Air Flow Analysis Example")
    print("=" * 60)
    
    # Run basic analysis
    cliff_map, data = basic_analysis_example()
    
    if data is not None:
        # Run visualization
        visualization_example(cliff_map, data)
        
        # Save results
        save_results_example(cliff_map, data)
        
        print("\n" + "=" * 60)
        print("AIR FLOW ANALYSIS COMPLETE!")
        print("=" * 60)
        print("✓ Analysis completed successfully")
        if HAS_MATPLOTLIB:
            print("✓ Check generated files: air_flow_analysis.png, air_flow_results.xml")
        else:
            print("✓ Check generated files: air_flow_results.xml, air_flow_components.csv")
    else:
        print("❌ Example failed - no data available")


if __name__ == "__main__":
    main()