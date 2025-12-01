#!/usr/bin/env python3
"""
Pedestrian Flow Analysis Example using CLiFF-map

This example demonstrates how to analyze pedestrian flow patterns using the enhanced CLiFF-map
algorithm, focusing on large datasets and parallel processing.

Dependencies (optional):
- pandas: for advanced data loading (fallback: synthetic data)
- matplotlib: for visualization (fallback: text output)  
- scikit-learn: for additional processing (fallback: basic algorithms)
"""

import sys
import os
import numpy as np
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for optional dependencies
HAS_PANDAS = False
HAS_MATPLOTLIB = False

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

# Import CLiFF-map
from cliffmap.dynamic_map import DynamicMap


def generate_pedestrian_data(n_points=5000, random_seed=42):
    """Generate synthetic pedestrian flow data with realistic movement patterns."""
    np.random.seed(random_seed)
    
    # Create spatial domain representing a plaza or walkway
    x_range = (-10, 10)
    y_range = (-8, 8)
    
    # Generate points
    x = np.random.uniform(x_range[0], x_range[1], n_points)
    y = np.random.uniform(y_range[0], y_range[1], n_points)
    
    # Create pedestrian flow patterns
    vx_list = []
    vy_list = []
    
    for xi, yi in zip(x, y):
        # Main flow patterns
        if yi > 5:  # Top area - flow right
            vx = 1.5 + np.random.normal(0, 0.3)
            vy = 0.2 * np.random.normal()
        elif yi < -5:  # Bottom area - flow left
            vx = -1.5 + np.random.normal(0, 0.3)
            vy = 0.2 * np.random.normal()
        elif abs(xi) < 2:  # Central corridor - vertical flows
            if np.random.random() < 0.5:
                vx = 0.3 * np.random.normal()
                vy = 2.0 + np.random.normal(0, 0.4)  # Up
            else:
                vx = 0.3 * np.random.normal()
                vy = -2.0 + np.random.normal(0, 0.4)  # Down
        else:  # Side areas - converging flows
            # Flow toward center
            target_x = 0
            target_y = 0
            dx = target_x - xi
            dy = target_y - yi
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist > 0:
                vx = (dx / dist) * (1.0 + np.random.normal(0, 0.2))
                vy = (dy / dist) * (1.0 + np.random.normal(0, 0.2))
            else:
                vx = np.random.normal(0, 0.1)
                vy = np.random.normal(0, 0.1)
        
        # Add random walking component
        vx += np.random.normal(0, 0.1)
        vy += np.random.normal(0, 0.1)
        
        vx_list.append(vx)
        vy_list.append(vy)
    
    # Create data array [x, y, vx, vy]
    data = np.column_stack([x, y, vx_list, vy_list])
    
    print(f"Generated {n_points} synthetic pedestrian data points")
    print(f"Spatial domain: x∈{x_range}, y∈{y_range}")
    speeds = np.sqrt(np.array(vx_list)**2 + np.array(vy_list)**2)
    print(f"Speed range: {np.min(speeds):.2f} - {np.max(speeds):.2f} m/s")
    
    return data


def load_pedestrian_data():
    """Load pedestrian data from file or generate synthetic data."""
    print("Loading pedestrian flow data...")
    
    # Try to load real data if available
    possible_paths = [
        "data/pedestrian_flow.csv",
        "../Data/pedestrian.csv",
        "../../pedestrian.csv"
    ]
    
    for data_file in possible_paths:
        if os.path.exists(data_file) and HAS_PANDAS:
            try:
                df = pd.read_csv(data_file)
                data = df.values
                print(f"Loaded real data from {data_file}: {data.shape}")
                return data
            except Exception as e:
                print(f"Failed to load {data_file}: {e}")
    
    # Generate synthetic data
    return generate_pedestrian_data()


def large_dataset_analysis():
    """Demonstrate analysis on large pedestrian dataset."""
    print("🚶  CLiFF-map Pedestrian Flow Analysis Example")
    print("=" * 60)
    print("LARGE DATASET ANALYSIS")
    print("=" * 60)
    
    # Load data
    data = load_pedestrian_data()
    if data is None:
        print("❌ Failed to load data")
        return None, None
        
    print(f"\nData shape: {data.shape}")
    print(f"Columns: {data.shape[1]} (expected: x, y, vx, vy)")
    
    # Create CLiFF-map optimized for larger datasets
    cliff_map = DynamicMap(
        batch_size=100,  # Larger batches for efficiency
        max_iterations=150,
        bandwidth=0.8,    # Slightly larger bandwidth for pedestrian data
        min_samples=10,
        parallel=True,
        n_jobs=4,         # Use more cores for larger dataset
        verbose=True,
        progress=True
    )
    
    print("\nFitting CLiFF-map model...")
    start_time = time.time()
    
    try:
        cliff_map.fit(data)
        fit_time = time.time() - start_time
        
        # Print results
        print(f"\nResults:")
        print(f"- Found {len(cliff_map.components)} pedestrian flow components")
        print(f"- Data points: {data.shape[0]}")
        print(f"- Processing time: {fit_time:.2f} seconds")
        print(f"- Points per second: {data.shape[0]/fit_time:.1f}")
        
        # Display component details
        print(f"\nComponent Details:")
        for i, component in enumerate(cliff_map.components):
            pos = component['position']
            velocity = np.array([component.get('velocity_x', 0), component.get('velocity_y', 0)])
            speed = np.linalg.norm(velocity)
            direction_deg = np.degrees(np.arctan2(velocity[1], velocity[0]))
            
            print(f"  Component {i+1}:")
            print(f"    Position: ({pos[0]:.2f}, {pos[1]:.2f})")
            print(f"    Velocity: ({velocity[0]:.2f}, {velocity[1]:.2f}) m/s")
            print(f"    Speed: {speed:.2f} m/s")
            print(f"    Direction: {direction_deg:.1f}°")
            print(f"    Weight: {component['weight']:.3f}")
    
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
    return cliff_map if 'cliff_map' in locals() else None, data


def visualization_example(cliff_map, data):
    """Demonstrate pedestrian flow visualization."""
    print("\n" + "=" * 60)
    print("PEDESTRIAN FLOW VISUALIZATION")
    print("=" * 60)
    
    if cliff_map is None:
        print("No CLiFF-map object available for visualization")
        return
        
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available - skipping visualization")
        return
        
    try:
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Data points colored by speed
        plt.subplot(1, 3, 1)
        if data.shape[1] >= 4:
            speeds = np.sqrt(data[:, 2]**2 + data[:, 3]**2)
            scatter = plt.scatter(data[:, 0], data[:, 1], c=speeds, 
                                cmap='viridis', alpha=0.6, s=5)
            plt.colorbar(scatter, label='Speed (m/s)')
        else:
            plt.scatter(data[:, 0], data[:, 1], alpha=0.6, s=5)
        plt.title('Pedestrian Speeds')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Flow components
        plt.subplot(1, 3, 2)
        plt.scatter(data[:, 0], data[:, 1], c='lightgray', alpha=0.2, s=2, label='Pedestrians')
        
        if cliff_map.components:
            for i, comp in enumerate(cliff_map.components):
                pos = comp['position']
                vx = comp.get('velocity_x', 0)
                vy = comp.get('velocity_y', 0)
                weight = comp['weight']
                
                # Plot component position
                plt.scatter(pos[0], pos[1], s=200*weight, c='red', 
                           alpha=0.8, edgecolors='black', zorder=5)
                
                # Plot velocity arrow
                plt.arrow(pos[0], pos[1], vx*2, vy*2, head_width=0.3, 
                         head_length=0.3, fc='blue', ec='blue', zorder=5)
        
        plt.title(f'Flow Components ({len(cliff_map.components)})')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 3: Speed distribution
        plt.subplot(1, 3, 3)
        if data.shape[1] >= 4:
            speeds = np.sqrt(data[:, 2]**2 + data[:, 3]**2)
            plt.hist(speeds, bins=30, alpha=0.7, color='green', edgecolor='black')
            plt.axvline(np.mean(speeds), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(speeds):.2f} m/s')
            plt.xlabel('Speed (m/s)')
            plt.ylabel('Count')
            plt.title('Speed Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pedestrian_flow_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved visualization as 'pedestrian_flow_analysis.png'")
        
    except Exception as e:
        print(f"Visualization failed: {e}")


def save_results_example(cliff_map, data):
    """Save pedestrian flow results."""
    print("\n" + "=" * 60)
    print("SAVING PEDESTRIAN FLOW RESULTS")
    print("=" * 60)
    
    if cliff_map is None:
        print("No CLiFF-map object available for saving")
        return
    
    try:
        # Save to XML
        xml_filename = 'pedestrian_flow_results.xml'
        cliff_map.save_xml(xml_filename, include_metadata=True)
        print(f"✓ Saved XML results: {xml_filename}")
        
        # Save to CSV
        csv_filename = 'pedestrian_components.csv'
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
    """Run pedestrian flow analysis example."""
    print("🚶  CLiFF-map Pedestrian Flow Analysis Example")
    print("=" * 60)
    
    # Run analysis
    cliff_map, data = large_dataset_analysis()
    
    if data is not None:
        # Run visualization
        visualization_example(cliff_map, data)
        
        # Save results
        save_results_example(cliff_map, data)
        
        print("\n" + "=" * 60)
        print("PEDESTRIAN FLOW ANALYSIS COMPLETE!")
        print("=" * 60)
        print("✓ Analysis completed successfully")
        if HAS_MATPLOTLIB:
            print("✓ Check generated files: pedestrian_flow_analysis.png")
        print("✓ Check generated files: pedestrian_flow_results.xml")
    else:
        print("❌ Example failed - no data available")


if __name__ == "__main__":
    main()