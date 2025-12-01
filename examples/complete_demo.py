#!/usr/bin/env python3
"""
Complete CLiFF-map Package Demonstration

This comprehensive example demonstrates all major features of the CLiFF-map
Python package including data loading, analysis, visualization, checkpointing,
and result export across multiple data types.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import warnings

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import CLiFF-map package
try:
    from cliffmap import (
        DynamicMap, FlowFieldVisualizer, CheckpointManager,
        get_version_info, print_version_info, DEFAULT_CONFIG
    )
    print("✓ CLiFF-map package imported successfully")
except ImportError as e:
    print(f"✗ Error importing CLiFF-map package: {e}")
    sys.exit(1)

import pandas as pd


def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print("  ____  _      _  __  __              ____                  _____  ")
    print(" / ___|| |    (_)/ _|/ _|     _ __ __| _   \\ _ __   __ _  |___  \\ ") 
    print("| |    | |    | | |_| |_     | '_  _ |  _ \\ | '_ \\ / _` | |   _/ /  ")
    print("| |    | |    | |  _|  _|    | | | | | |_) | | | | (_| | |  (_> <   ")
    print(" \\____|_|____/|_|_| |_|      |_| |_| |____/|_| |_|\\__,_|  \\___/\\_\\  ")
    print("")
    print("           Circular-Linear Flow Field Mapping for Python")
    print("                  Complete Package Demonstration")
    print("=" * 80)


def check_dependencies():
    """Check and display package dependencies."""
    print("\n📋 DEPENDENCY CHECK")
    print("-" * 50)
    
    print_version_info()
    
    # Check for optional dependencies
    optional_deps = {
        'tqdm': 'Progress monitoring',
        'seaborn': 'Enhanced visualization',
        'joblib': 'Parallel processing'
    }
    
    print(f"\nOptional dependencies:")
    for package, description in optional_deps.items():
        try:
            __import__(package)
            print(f"  ✓ {package}: {description}")
        except ImportError:
            print(f"  ⚠ {package}: {description} (not available)")


def discover_available_data():
    """Discover available data files for analysis."""
    print(f"\n📂 DATA DISCOVERY")
    print("-" * 50)
    
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    data_files = {}
    
    # Look for specific data files
    target_files = [
        ('air_flow.csv', 'Air flow sensor data'),
        ('pedestrian.csv', 'Pedestrian movement data'),
        ('atc/atc-20121104-10_nh.csv', 'Traffic counter data (sample)')
    ]
    
    for file_pattern, description in target_files:
        file_path = os.path.join(base_dir, file_pattern)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB
            try:
                df = pd.read_csv(file_path)
                data_files[file_pattern] = {
                    'path': file_path,
                    'description': description,
                    'size_kb': file_size,
                    'points': len(df),
                    'columns': df.shape[1]
                }
                print(f"  ✓ {file_pattern}: {description}")
                print(f"    Size: {file_size:.1f} KB, Points: {len(df)}, Columns: {df.shape[1]}")
            except Exception as e:
                print(f"  ⚠ {file_pattern}: Could not read ({e})")
        else:
            print(f"  ✗ {file_pattern}: Not found")
    
    return data_files


def create_synthetic_data():
    """Create synthetic flow data for demonstration."""
    print(f"\n🔧 CREATING SYNTHETIC DATA")
    print("-" * 50)
    
    np.random.seed(42)
    
    # Create realistic multi-modal flow data
    n_total = 1000
    
    # Main corridor flow (eastward)
    n_main = n_total // 3
    main_positions = np.random.multivariate_normal([5, 5], [[0.3, 0], [0, 1.0]], n_main)
    main_directions = np.random.normal(0, 0.3, n_main)  # Eastward
    main_speeds = np.random.lognormal(0, 0.3, n_main)
    
    # Cross flow (northward) 
    n_cross = n_total // 3
    cross_positions = np.random.multivariate_normal([5, 2], [[1.0, 0], [0, 0.3]], n_cross)
    cross_directions = np.random.normal(np.pi/2, 0.4, n_cross)  # Northward
    cross_speeds = np.random.lognormal(-0.2, 0.4, n_cross)
    
    # Local circulation/turbulence
    n_local = n_total - n_main - n_cross
    local_positions = np.random.multivariate_normal([8, 8], [[0.5, 0.2], [0.2, 0.5]], n_local)
    local_directions = np.random.uniform(0, 2*np.pi, n_local)  # Random directions
    local_speeds = np.random.exponential(0.5, n_local)
    
    # Combine all components
    positions = np.vstack([main_positions, cross_positions, local_positions])
    directions = np.hstack([main_directions, cross_directions, local_directions])
    speeds = np.hstack([main_speeds, cross_speeds, local_speeds])
    
    # Create final data array
    synthetic_data = np.column_stack([positions, directions, speeds])
    
    print(f"  Created {len(synthetic_data)} synthetic data points")
    print(f"  Components: {n_main} main flow, {n_cross} cross flow, {n_local} local circulation")
    print(f"  Data shape: {synthetic_data.shape}")
    
    return synthetic_data


def analyze_dataset(data, dataset_name, output_dir):
    """
    Perform comprehensive analysis on a dataset.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data array
    dataset_name : str
        Name of the dataset for output files
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    dict : Analysis results
    """
    print(f"\n🔬 ANALYZING: {dataset_name.upper()}")
    print("-" * 50)
    
    # Create dataset-specific output directory
    dataset_dir = os.path.join(output_dir, f"{dataset_name}_analysis")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Initialize CLiFF-map with comprehensive settings
    print("  Initializing CLiFF-map...")
    
    cliff_map = DynamicMap(
        batch_size=min(100, len(data) // 5),  # Adaptive batch size
        max_iterations=200,
        bandwidth=determine_bandwidth(data),
        min_samples=max(5, len(data) // 200),  # Adaptive minimum samples
        parallel=True,
        n_jobs=2,
        verbose=True,
        progress=True
    )
    
    print(f"    Batch size: {cliff_map.batch_size}")
    print(f"    Bandwidth: {cliff_map.bandwidth:.3f}")
    print(f"    Min samples: {cliff_map.min_samples}")
    
    # Perform analysis with timing
    print(f"  Running flow field analysis...")
    start_time = time.time()
    
    try:
        cliff_map.fit(data)
        analysis_time = time.time() - start_time
        
        print(f"    ✓ Analysis completed in {analysis_time:.2f} seconds")
        print(f"    Found {len(cliff_map.components)} flow components")
        
    except Exception as e:
        print(f"    ✗ Analysis failed: {e}")
        return None
    
    # Create visualizations
    print(f"  Creating visualizations...")
    
    try:
        visualizer = FlowFieldVisualizer(figsize=(15, 10))
        
        # Component visualization
        fig, ax = visualizer.plot_components(
            cliff_map,
            save_path=os.path.join(dataset_dir, f"{dataset_name}_components.png"),
            show_ellipses=True
        )
        plt.close(fig)
        
        # Flow field visualization
        fig, ax = visualizer.plot_flow_field(
            cliff_map,
            resolution=40,
            save_path=os.path.join(dataset_dir, f"{dataset_name}_flow_field.png")
        )
        plt.close(fig)
        
        # Training history (if available)
        if hasattr(cliff_map, 'history') and cliff_map.history:
            fig, axes = visualizer.plot_training_history(
                cliff_map,
                save_path=os.path.join(dataset_dir, f"{dataset_name}_training.png")
            )
            plt.close(fig)
        
        print(f"    ✓ Visualizations saved to {dataset_dir}/")
        
    except Exception as e:
        print(f"    ⚠ Visualization warning: {e}")
    
    # Save checkpoint
    print(f"  Creating checkpoint...")
    
    try:
        checkpoint_manager = CheckpointManager(os.path.join(dataset_dir, "checkpoints"))
        
        metadata = {
            'dataset_name': dataset_name,
            'data_points': len(data),
            'analysis_time': analysis_time,
            'n_components': len(cliff_map.components),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        checkpoint_path = checkpoint_manager.save_checkpoint(
            cliff_map,
            checkpoint_name=f"{dataset_name}_analysis",
            metadata=metadata
        )
        
        print(f"    ✓ Checkpoint saved: {os.path.basename(checkpoint_path)}")
        
    except Exception as e:
        print(f"    ⚠ Checkpointing warning: {e}")
        checkpoint_path = None
    
    # Export results
    print(f"  Exporting results...")
    
    try:
        # Component details to CSV
        components_data = []
        for i, comp in enumerate(cliff_map.components):
            components_data.append({
                'component_id': i + 1,
                'x_position': comp['position'][0],
                'y_position': comp['position'][1],
                'direction_rad': comp['direction'],
                'direction_deg': np.degrees(comp['direction']) % 360,
                'weight': comp['weight'],
                'uncertainty': comp.get('uncertainty', 0)
            })
        
        df_components = pd.DataFrame(components_data)
        csv_path = os.path.join(dataset_dir, f"{dataset_name}_components.csv")
        df_components.to_csv(csv_path, index=False)
        
        print(f"    ✓ Component data exported: {os.path.basename(csv_path)}")
        
    except Exception as e:
        print(f"    ⚠ Export warning: {e}")
    
    # Prepare results summary
    results = {
        'dataset_name': dataset_name,
        'cliff_map': cliff_map,
        'data': data,
        'analysis_time': analysis_time,
        'n_components': len(cliff_map.components),
        'output_directory': dataset_dir,
        'checkpoint_path': checkpoint_path
    }
    
    return results


def determine_bandwidth(data):
    """Determine optimal bandwidth for the dataset."""
    # Simple heuristic based on data spread
    spatial_data = data[:, :2]
    
    # Calculate inter-quartile range
    q75, q25 = np.percentile(spatial_data, [75, 25], axis=0)
    iqr = np.mean(q75 - q25)
    
    # Scott's rule adaptation
    n = len(data)
    bandwidth = 1.06 * iqr * (n ** (-1/5))
    
    # Ensure reasonable bounds
    bandwidth = np.clip(bandwidth, 0.1, 3.0)
    
    return bandwidth


def comparative_analysis(results_list):
    """Create comparative analysis across datasets."""
    if len(results_list) < 2:
        print("  Need at least 2 datasets for comparison")
        return
    
    print(f"\n📊 COMPARATIVE ANALYSIS")
    print("-" * 50)
    
    output_dir = "complete_demo_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison visualization
    print("  Creating comparison visualizations...")
    
    try:
        visualizer = FlowFieldVisualizer(figsize=(18, 6))
        
        # Extract CLiFF-maps for comparison
        cliff_maps = {result['dataset_name']: result['cliff_map'] 
                     for result in results_list}
        
        fig, axes = visualizer.compare_results(
            cliff_maps,
            save_path=os.path.join(output_dir, "dataset_comparison.png")
        )
        plt.close(fig)
        
        print(f"    ✓ Comparison saved to {output_dir}/dataset_comparison.png")
        
    except Exception as e:
        print(f"    ⚠ Comparison visualization warning: {e}")
    
    # Create summary statistics
    print("  Generating summary statistics...")
    
    summary_data = []
    for result in results_list:
        summary_data.append({
            'dataset': result['dataset_name'],
            'data_points': len(result['data']),
            'components': result['n_components'],
            'analysis_time_sec': result['analysis_time'],
            'components_per_1000_points': result['n_components'] / (len(result['data']) / 1000),
            'processing_rate_points_per_sec': len(result['data']) / result['analysis_time']
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "analysis_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    
    print(f"    ✓ Summary statistics saved to {summary_path}")
    
    # Print summary table
    print(f"\n  Summary Results:")
    print(f"  {'Dataset':<15} {'Points':<8} {'Components':<11} {'Time (s)':<10} {'Rate (pts/s)':<12}")
    print(f"  {'-'*15:<15} {'-'*8:<8} {'-'*11:<11} {'-'*10:<10} {'-'*12:<12}")
    
    for _, row in df_summary.iterrows():
        print(f"  {row['dataset']:<15} {row['data_points']:<8} {row['components']:<11} "
              f"{row['analysis_time_sec']:<10.2f} {row['processing_rate_points_per_sec']:<12.0f}")


def demonstrate_checkpointing():
    """Demonstrate advanced checkpointing features."""
    print(f"\n💾 CHECKPOINTING DEMONSTRATION")
    print("-" * 50)
    
    output_dir = "complete_demo_results/checkpointing_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test data
    test_data = create_synthetic_data()[:500]  # Smaller dataset for demo
    
    print("  Creating training session with checkpointing...")
    
    try:
        from cliffmap.checkpoint import create_training_session
        
        # Create session manager
        session_manager = create_training_session("demo_session", output_dir)
        
        # Initialize CLiFF-map for long training
        cliff_map = DynamicMap(
            batch_size=50,
            max_iterations=100,
            bandwidth=0.5,
            verbose=False,
            progress=True
        )
        
        # Save initial checkpoint
        initial_metadata = {'stage': 'initialization', 'data_points': len(test_data)}
        session_manager.save_checkpoint(
            cliff_map,
            checkpoint_name="initial",
            metadata=initial_metadata
        )
        
        # Simulate partial training
        print("  Running partial analysis...")
        cliff_map.fit(test_data[:200])  # Partial data
        
        # Save intermediate checkpoint
        intermediate_metadata = {'stage': 'intermediate', 'data_processed': 200}
        checkpoint_path = session_manager.save_checkpoint(
            cliff_map,
            checkpoint_name="intermediate", 
            metadata=intermediate_metadata
        )
        
        print("  Simulating training interruption and resumption...")
        
        # Load checkpoint and continue
        restored_map, metadata = session_manager.load_checkpoint(checkpoint_path)
        
        # Complete training
        restored_map.fit(test_data)  # Full data
        
        # Final checkpoint
        final_metadata = {'stage': 'completed', 'final_components': len(restored_map.components)}
        session_manager.save_checkpoint(
            restored_map,
            checkpoint_name="final",
            metadata=final_metadata
        )
        
        # Show checkpoint information
        print(f"  Checkpoint session summary:")
        checkpoints = session_manager.list_checkpoints()
        for cp in checkpoints:
            info = session_manager.get_checkpoint_info(cp)
            stage = info['metadata'].get('stage', 'unknown')
            components = info.get('n_components', 0)
            size_mb = info.get('file_size_mb', 0)
            print(f"    - {os.path.basename(cp)}: {stage} stage, {components} components, {size_mb:.1f} MB")
        
        print(f"    ✓ Checkpointing demonstration completed successfully")
        
    except Exception as e:
        print(f"    ✗ Checkpointing demonstration failed: {e}")


def performance_benchmark():
    """Run performance benchmarks."""
    print(f"\n⚡ PERFORMANCE BENCHMARK")
    print("-" * 50)
    
    # Test different data sizes
    sizes = [100, 500, 1000, 2000]
    results = []
    
    print("  Testing performance across different data sizes...")
    
    for size in sizes:
        print(f"    Testing with {size} points...")
        
        # Generate test data
        data = create_synthetic_data()[:size]
        
        # Configure for speed
        cliff_map = DynamicMap(
            batch_size=min(50, size // 4),
            max_iterations=50,
            bandwidth=0.5,
            min_samples=5,
            parallel=True,
            n_jobs=2,
            verbose=False,
            progress=False
        )
        
        # Time the analysis
        start_time = time.time()
        cliff_map.fit(data)
        processing_time = time.time() - start_time
        
        results.append({
            'size': size,
            'time': processing_time,
            'components': len(cliff_map.components),
            'rate': size / processing_time
        })
        
        print(f"      {processing_time:.2f}s, {len(cliff_map.components)} components, {size/processing_time:.0f} pts/s")
    
    # Find best performance
    best_rate = max(results, key=lambda x: x['rate'])
    print(f"    Best performance: {best_rate['rate']:.0f} points/second at {best_rate['size']} points")
    
    return results


def main():
    """Run complete package demonstration."""
    print_banner()
    
    # 1. Dependency check
    check_dependencies()
    
    # 2. Data discovery
    available_data = discover_available_data()
    
    # 3. Create output directory
    output_dir = "complete_demo_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Analyze available datasets
    results = []
    
    # Always analyze synthetic data
    print(f"\n🧪 SYNTHETIC DATA ANALYSIS")
    print("=" * 50)
    synthetic_data = create_synthetic_data()
    synthetic_result = analyze_dataset(synthetic_data, "synthetic", output_dir)
    if synthetic_result:
        results.append(synthetic_result)
    
    # Analyze real datasets if available
    for file_pattern, file_info in available_data.items():
        try:
            print(f"\n🗂️  REAL DATA ANALYSIS: {file_info['description'].upper()}")
            print("=" * 50)
            
            data = pd.read_csv(file_info['path']).values
            
            # Limit size for demonstration
            if len(data) > 3000:
                indices = np.random.choice(len(data), 3000, replace=False)
                data = data[indices]
                print(f"  Sampled {len(data)} points from {file_info['points']} total")
            
            dataset_name = os.path.splitext(os.path.basename(file_pattern))[0]
            result = analyze_dataset(data, dataset_name, output_dir)
            
            if result:
                results.append(result)
                
        except Exception as e:
            print(f"  ⚠ Could not analyze {file_pattern}: {e}")
    
    # 5. Comparative analysis
    if len(results) > 1:
        comparative_analysis(results)
    
    # 6. Advanced features demonstration
    demonstrate_checkpointing()
    
    # 7. Performance benchmark
    performance_results = performance_benchmark()
    
    # 8. Final summary
    print(f"\n🎯 DEMONSTRATION COMPLETE")
    print("=" * 50)
    print(f"Analyzed {len(results)} datasets successfully:")
    for result in results:
        print(f"  ✓ {result['dataset_name']}: {result['n_components']} components from {len(result['data'])} points")
    
    print(f"\n📁 All results saved to: {output_dir}/")
    print(f"📊 Package features demonstrated:")
    print(f"  ✓ Multi-dataset analysis")
    print(f"  ✓ Parallel processing")
    print(f"  ✓ Progress monitoring")
    print(f"  ✓ Comprehensive visualization")
    print(f"  ✓ Checkpointing and state management")
    print(f"  ✓ Result export and comparison")
    print(f"  ✓ Performance benchmarking")
    
    print(f"\n🚀 CLiFF-map Python package demonstration completed successfully!")


if __name__ == "__main__":
    main()