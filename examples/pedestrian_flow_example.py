#!/usr/bin/env python3
"""
Pedestrian Flow Analysis Example

This example demonstrates CLiFF-map analysis on pedestrian flow data,
focusing on large dataset handling, parallel processing, and advanced features.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cliffmap import DynamicMap, FlowFieldVisualizer, CheckpointManager
from cliffmap.checkpoint import create_training_session, auto_checkpoint
import pandas as pd


def load_pedestrian_data():
    """Load pedestrian flow data from CSV file."""
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'pedestrian.csv')
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please ensure pedestrian.csv exists in the project root directory")
        return None
    
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded pedestrian data: {df.shape[0]} points, {df.shape[1]} dimensions")
        
        # Display data info
        print(f"Data columns: {list(df.columns)}")
        print(f"Data range:")
        for col in df.columns:
            print(f"  {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")
        
        return df.values
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def large_dataset_analysis():
    """Demonstrate analysis on large pedestrian dataset with progress monitoring."""
    print("=" * 60)
    print("LARGE DATASET PEDESTRIAN ANALYSIS")
    print("=" * 60)
    
    # Load data
    data = load_pedestrian_data()
    if data is None:
        return None
    
    # Initialize with optimal settings for large datasets
    cliff_map = DynamicMap(
        batch_size=100,  # Larger batch size for efficiency
        max_iterations=200,
        bandwidth=0.4,
        min_samples=10,  # Higher threshold for noise reduction
        parallel=True,
        n_jobs=2,
        verbose=True,
        progress=True
    )
    
    print(f"\nProcessing {data.shape[0]} pedestrian data points...")
    print("Features:")
    print("- Large batch size for efficiency")
    print("- Parallel processing enabled")
    print("- Progress monitoring with tqdm")
    print("- Higher min_samples for noise reduction")
    
    # Measure processing time
    start_time = time.time()
    cliff_map.fit(data)
    processing_time = time.time() - start_time
    
    # Print results
    print(f"\nProcessing Results:")
    print(f"- Total processing time: {processing_time:.2f} seconds")
    print(f"- Found {len(cliff_map.components)} pedestrian flow components")
    print(f"- Average time per data point: {processing_time/data.shape[0]*1000:.3f} ms")
    
    # Component analysis
    print(f"\nPedestrian Flow Components:")
    for i, component in enumerate(cliff_map.components):
        pos = component['position']
        direction_deg = np.degrees(component['direction']) % 360
        print(f"  Flow {i+1}:")
        print(f"    Location: ({pos[0]:.2f}, {pos[1]:.2f})")
        print(f"    Direction: {direction_deg:.1f}° ({'N' if 315 <= direction_deg or direction_deg < 45 else 'E' if 45 <= direction_deg < 135 else 'S' if 135 <= direction_deg < 225 else 'W'})")
        print(f"    Strength: {component['weight']:.3f}")
    
    return cliff_map, data, processing_time


def parallel_processing_benchmark(data):
    """Benchmark different parallel processing configurations."""
    print("\n" + "=" * 60)
    print("PARALLEL PROCESSING BENCHMARK")
    print("=" * 60)
    
    configurations = [
        {'parallel': False, 'n_jobs': 1, 'name': 'Sequential'},
        {'parallel': True, 'n_jobs': 1, 'name': 'Parallel (1 core)'},
        {'parallel': True, 'n_jobs': 2, 'name': 'Parallel (2 cores)'},
        {'parallel': True, 'n_jobs': 4, 'name': 'Parallel (4 cores)'}
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\nTesting: {config['name']}")
        
        cliff_map = DynamicMap(
            batch_size=80,
            max_iterations=100,
            bandwidth=0.4,
            min_samples=8,
            parallel=config['parallel'],
            n_jobs=config['n_jobs'],
            verbose=False,
            progress=False  # Disable for timing accuracy
        )
        
        start_time = time.time()
        cliff_map.fit(data)
        processing_time = time.time() - start_time
        
        results[config['name']] = {
            'time': processing_time,
            'components': len(cliff_map.components),
            'cliff_map': cliff_map
        }
        
        print(f"  Time: {processing_time:.2f}s, Components: {len(cliff_map.components)}")
    
    # Find best configuration
    best_config = min(results.keys(), key=lambda k: results[k]['time'])
    speedup = results['Sequential']['time'] / results[best_config]['time']
    
    print(f"\nBenchmark Results:")
    print(f"- Best configuration: {best_config}")
    print(f"- Speedup: {speedup:.2f}x")
    print(f"- Time savings: {results['Sequential']['time'] - results[best_config]['time']:.2f}s")
    
    return results


def checkpointing_with_interruption_example(data):
    """Demonstrate checkpointing with simulated training interruption."""
    print("\n" + "=" * 60)
    print("CHECKPOINTING WITH INTERRUPTION SIMULATION")
    print("=" * 60)
    
    # Create training session
    session_manager = create_training_session("pedestrian_analysis")
    
    print("\n1. Starting training with automatic checkpointing...")
    
    # Initialize for longer training (to demonstrate checkpointing)
    cliff_map = DynamicMap(
        batch_size=50,
        max_iterations=300,  # Longer training
        bandwidth=0.4,
        min_samples=8,
        parallel=True,
        n_jobs=2,
        verbose=True,
        progress=True
    )
    
    # Simulate partial training with checkpoints
    print("   Training will be checkpointed every 20 iterations...")
    
    # Custom training loop with checkpointing
    cliff_map.data = data
    cliff_map.components = []
    
    try:
        # Simulate training with automatic checkpoints
        checkpoint_path = session_manager.save_checkpoint(
            cliff_map,
            checkpoint_name="initial_state",
            metadata={'iteration': 0, 'stage': 'initial'}
        )
        
        # Run partial training
        cliff_map.fit(data[:1000])  # Simulate partial processing
        
        # Save intermediate checkpoint
        intermediate_path = session_manager.save_checkpoint(
            cliff_map,
            checkpoint_name="intermediate_state",
            metadata={'iteration': 100, 'stage': 'intermediate', 'data_points': 1000}
        )
        
        print("\n2. Simulating training interruption...")
        print("   (In real scenario, this could be system crash, power failure, etc.)")
        
        print("\n3. Resuming from checkpoint...")
        
        # Resume training
        restored_map, metadata = session_manager.load_checkpoint(intermediate_path)
        
        # Continue with full dataset
        restored_map.fit(data)
        
        # Final checkpoint
        final_path = session_manager.save_checkpoint(
            restored_map,
            checkpoint_name="final_state",
            metadata={'iteration': 200, 'stage': 'completed', 'components': len(restored_map.components)}
        )
        
        print(f"   Training completed with {len(restored_map.components)} components")
        
        # Show checkpoint information
        print("\n4. Checkpoint summary:")
        checkpoints = session_manager.list_checkpoints()
        for cp in checkpoints:
            info = session_manager.get_checkpoint_info(cp)
            print(f"  - {os.path.basename(cp)}")
            print(f"    Stage: {info['metadata'].get('stage', 'Unknown')}")
            print(f"    Components: {info.get('n_components', 0)}")
            print(f"    Size: {info.get('file_size_mb', 0):.1f} MB")
        
        return restored_map
        
    except Exception as e:
        print(f"Error in checkpointing example: {e}")
        return None


def advanced_visualization_example(cliff_map, data):
    """Create advanced visualizations for pedestrian flow."""
    print("\n" + "=" * 60)
    print("ADVANCED PEDESTRIAN FLOW VISUALIZATIONS")
    print("=" * 60)
    
    # Create output directory
    output_dir = "pedestrian_flow_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = FlowFieldVisualizer(figsize=(16, 6))
    
    # 1. Comprehensive flow analysis
    print("\n1. Creating comprehensive flow field visualization...")
    fig, ax = visualizer.plot_flow_field(
        cliff_map,
        resolution=40,
        save_path=os.path.join(output_dir, "pedestrian_flow_field.png")
    )
    
    # 2. Component analysis with detailed view
    print("2. Creating detailed component analysis...")
    fig, ax = visualizer.plot_components(
        cliff_map,
        save_path=os.path.join(output_dir, "pedestrian_components.png"),
        show_ellipses=True
    )
    
    # 3. Training convergence analysis
    if hasattr(cliff_map, 'history') and cliff_map.history:
        print("3. Creating training convergence analysis...")
        fig, axes = visualizer.plot_training_history(
            cliff_map,
            save_path=os.path.join(output_dir, "pedestrian_training_history.png")
        )
    
    # 4. Data distribution with directional analysis
    print("4. Creating data distribution analysis...")
    from cliffmap.visualization import plot_data_distribution
    
    fig = plot_data_distribution(
        data,
        save_path=os.path.join(output_dir, "pedestrian_data_distribution.png"),
        bins=60
    )
    
    # 5. Flow intensity heatmap
    print("5. Creating flow intensity heatmap...")
    create_flow_intensity_heatmap(cliff_map, data, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}/")


def create_flow_intensity_heatmap(cliff_map, data, output_dir):
    """Create a heatmap showing flow intensity across the spatial domain."""
    
    # Create grid for heatmap
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    
    x_grid = np.linspace(x_min, x_max, 50)
    y_grid = np.linspace(y_min, y_max, 50)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Calculate flow intensity at each grid point
    intensity = np.zeros_like(xx)
    
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = np.array([xx[i, j], yy[i, j]])
            
            total_influence = 0
            for component in cliff_map.components:
                pos = component['position']
                weight = component['weight']
                dist = np.linalg.norm(point - pos)
                influence = weight * np.exp(-dist**2 / 1.0)  # Gaussian influence
                total_influence += influence
            
            intensity[i, j] = total_influence
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Plot heatmap
    contour = plt.contourf(xx, yy, intensity, levels=20, cmap='YlOrRd', alpha=0.8)
    plt.colorbar(contour, label='Flow Intensity')
    
    # Overlay data points
    plt.scatter(data[:, 0], data[:, 1], alpha=0.1, s=1, c='black')
    
    # Overlay components
    for i, component in enumerate(cliff_map.components):
        pos = component['position']
        plt.scatter(pos[0], pos[1], s=200, c='blue', marker='x', linewidth=3)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Pedestrian Flow Intensity Heatmap')
    plt.grid(True, alpha=0.3)
    
    heatmap_path = os.path.join(output_dir, "pedestrian_flow_intensity.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Flow intensity heatmap saved to: {heatmap_path}")


def export_detailed_results(cliff_map, data, processing_time):
    """Export comprehensive analysis results."""
    print("\n" + "=" * 60)
    print("EXPORTING DETAILED RESULTS")
    print("=" * 60)
    
    output_dir = "pedestrian_flow_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Component details
    components_data = []
    for i, comp in enumerate(cliff_map.components):
        pos = comp['position']
        direction = comp['direction']
        
        components_data.append({
            'component_id': i + 1,
            'x_position': pos[0],
            'y_position': pos[1],
            'direction_rad': direction,
            'direction_deg': np.degrees(direction) % 360,
            'weight': comp['weight'],
            'uncertainty': comp.get('uncertainty', 0),
            'flow_type': classify_flow_direction(direction)
        })
    
    df_components = pd.DataFrame(components_data)
    components_path = os.path.join(output_dir, "pedestrian_components_detailed.csv")
    df_components.to_csv(components_path, index=False)
    
    # 2. Analysis summary
    summary = {
        'dataset_info': {
            'total_points': len(data),
            'dimensions': data.shape[1],
            'spatial_extent': {
                'x_range': [float(data[:, 0].min()), float(data[:, 0].max())],
                'y_range': [float(data[:, 1].min()), float(data[:, 1].max())]
            }
        },
        'analysis_results': {
            'n_components': len(cliff_map.components),
            'processing_time_seconds': processing_time,
            'components_per_1000_points': len(cliff_map.components) / (len(data) / 1000)
        },
        'flow_analysis': {
            'dominant_direction': classify_flow_direction(
                np.angle(np.sum([comp['weight'] * np.exp(1j * comp['direction']) 
                               for comp in cliff_map.components]))
            ),
            'flow_diversity': calculate_flow_diversity(cliff_map.components),
            'average_component_weight': float(np.mean([c['weight'] for c in cliff_map.components]))
        }
    }
    
    summary_path = os.path.join(output_dir, "pedestrian_analysis_summary.json")
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Detailed components exported to: {components_path}")
    print(f"Analysis summary exported to: {summary_path}")
    
    return df_components, summary


def classify_flow_direction(direction_rad):
    """Classify flow direction into cardinal/intercardinal directions."""
    direction_deg = np.degrees(direction_rad) % 360
    
    if direction_deg < 22.5 or direction_deg >= 337.5:
        return "East"
    elif 22.5 <= direction_deg < 67.5:
        return "Northeast"
    elif 67.5 <= direction_deg < 112.5:
        return "North"
    elif 112.5 <= direction_deg < 157.5:
        return "Northwest"
    elif 157.5 <= direction_deg < 202.5:
        return "West"
    elif 202.5 <= direction_deg < 247.5:
        return "Southwest"
    elif 247.5 <= direction_deg < 292.5:
        return "South"
    else:
        return "Southeast"


def calculate_flow_diversity(components):
    """Calculate flow direction diversity using circular statistics."""
    if not components:
        return 0.0
    
    # Calculate resultant vector length (measure of direction concentration)
    weights = np.array([c['weight'] for c in components])
    directions = np.array([c['direction'] for c in components])
    
    # Weighted circular mean
    resultant_x = np.sum(weights * np.cos(directions))
    resultant_y = np.sum(weights * np.sin(directions))
    resultant_length = np.sqrt(resultant_x**2 + resultant_y**2) / np.sum(weights)
    
    # Diversity is inverse of concentration (1 - resultant_length)
    diversity = 1.0 - resultant_length
    
    return float(diversity)


def main():
    """Run complete pedestrian flow analysis example."""
    print("CLiFF-map Pedestrian Flow Analysis Example")
    print("==========================================")
    
    try:
        # 1. Large dataset analysis
        cliff_map, data, processing_time = large_dataset_analysis()
        if cliff_map is None:
            return
        
        # 2. Parallel processing benchmark
        benchmark_results = parallel_processing_benchmark(data[:2000])  # Use subset for faster benchmarking
        
        # 3. Checkpointing demonstration
        checkpointed_map = checkpointing_with_interruption_example(data[:1500])
        
        # 4. Advanced visualizations
        advanced_visualization_example(cliff_map, data)
        
        # 5. Export detailed results
        components_df, summary = export_detailed_results(cliff_map, data, processing_time)
        
        print("\n" + "=" * 60)
        print("PEDESTRIAN ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"- Processed {len(data)} pedestrian data points")
        print(f"- Identified {len(cliff_map.components)} flow components")
        print(f"- Processing time: {processing_time:.2f} seconds")
        print("- All results saved to 'pedestrian_flow_results' directory")
        print("- Checkpoints saved to 'pedestrian_analysis' directory")
        
        # Performance summary
        best_config = min(benchmark_results.keys(), key=lambda k: benchmark_results[k]['time'])
        print(f"- Best performance: {best_config}")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()