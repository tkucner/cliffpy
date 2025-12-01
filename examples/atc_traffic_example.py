#!/usr/bin/env python3
"""
ATC (Automated Traffic Counter) Data Analysis Example

This example demonstrates CLiFF-map analysis on traffic flow data,
focusing on temporal analysis, data preprocessing, and batch processing.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
from datetime import datetime
import time

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cliffmap import DynamicMap, FlowFieldVisualizer, CheckpointManager
import pandas as pd


def discover_atc_files():
    """Discover available ATC data files."""
    atc_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'atc')
    
    if not os.path.exists(atc_dir):
        print(f"ATC directory not found: {atc_dir}")
        return []
    
    # Find all ATC CSV files
    atc_files = glob.glob(os.path.join(atc_dir, "*.csv"))
    
    if not atc_files:
        print(f"No ATC CSV files found in: {atc_dir}")
        return []
    
    print(f"Found {len(atc_files)} ATC data files:")
    for f in sorted(atc_files)[:10]:  # Show first 10
        file_size = os.path.getsize(f) / 1024  # KB
        print(f"  - {os.path.basename(f)} ({file_size:.1f} KB)")
    
    if len(atc_files) > 10:
        print(f"  ... and {len(atc_files) - 10} more files")
    
    return sorted(atc_files)


def load_atc_data(file_path, max_points=None):
    """
    Load and preprocess ATC data.
    
    Parameters:
    -----------
    file_path : str
        Path to ATC CSV file
    max_points : int, optional
        Maximum number of points to load (for testing)
    
    Returns:
    --------
    np.ndarray : Processed data array
    dict : Metadata about the file
    """
    try:
        # Try to read the file
        df = pd.read_csv(file_path)
        
        if df.empty:
            print(f"Warning: Empty file {file_path}")
            return None, None
        
        print(f"Raw data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Basic preprocessing - ensure we have at least x, y coordinates
        if df.shape[1] < 2:
            print(f"Error: Insufficient columns in {file_path}")
            return None, None
        
        # Use first two columns as x, y coordinates
        data = df.iloc[:, :2].values
        
        # Add synthetic direction if only position data
        if df.shape[1] == 2:
            # Calculate movement direction from position differences
            directions = np.zeros(len(data))
            for i in range(1, len(data)):
                dx = data[i, 0] - data[i-1, 0]
                dy = data[i, 1] - data[i-1, 1]
                directions[i] = np.arctan2(dy, dx)
            
            # Add direction column
            data = np.column_stack([data, directions])
        
        # Add synthetic speed/magnitude if available
        if df.shape[1] >= 3 and df.shape[1] < 4:
            speeds = np.ones(len(data))  # Default speed
            data = np.column_stack([data, speeds])
        elif df.shape[1] >= 4:
            data = df.iloc[:, :4].values
        
        # Limit data if requested
        if max_points and len(data) > max_points:
            indices = np.random.choice(len(data), max_points, replace=False)
            data = data[indices]
            print(f"Sampled {max_points} points from {len(df)} total")
        
        # Remove invalid data points
        valid_mask = np.isfinite(data).all(axis=1)
        data = data[valid_mask]
        
        metadata = {
            'filename': os.path.basename(file_path),
            'original_points': len(df),
            'processed_points': len(data),
            'columns': list(df.columns),
            'file_size_kb': os.path.getsize(file_path) / 1024,
            'data_range': {
                'x': [float(data[:, 0].min()), float(data[:, 0].max())],
                'y': [float(data[:, 1].min()), float(data[:, 1].max())]
            }
        }
        
        print(f"Processed {len(data)} valid data points")
        return data, metadata
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def analyze_single_atc_file(file_path):
    """Analyze a single ATC data file."""
    print("=" * 60)
    print(f"ANALYZING: {os.path.basename(file_path)}")
    print("=" * 60)
    
    # Load data
    data, metadata = load_atc_data(file_path, max_points=2000)
    if data is None:
        return None
    
    print(f"\nDataset Information:")
    print(f"- File: {metadata['filename']}")
    print(f"- Points: {metadata['processed_points']} (from {metadata['original_points']} original)")
    print(f"- Spatial range: X=[{metadata['data_range']['x'][0]:.2f}, {metadata['data_range']['x'][1]:.2f}], Y=[{metadata['data_range']['y'][0]:.2f}, {metadata['data_range']['y'][1]:.2f}]")
    print(f"- File size: {metadata['file_size_kb']:.1f} KB")
    
    # Configure CLiFF-map for traffic data
    cliff_map = DynamicMap(
        batch_size=100,
        max_iterations=150,
        bandwidth=determine_optimal_bandwidth(data),
        min_samples=8,
        parallel=True,
        n_jobs=2,
        verbose=True,
        progress=True
    )
    
    print(f"\nRunning CLiFF-map analysis...")
    print(f"- Bandwidth: {cliff_map.bandwidth}")
    print(f"- Batch size: {cliff_map.batch_size}")
    print(f"- Parallel processing: {cliff_map.parallel}")
    
    # Analyze
    start_time = time.time()
    cliff_map.fit(data)
    analysis_time = time.time() - start_time
    
    # Results
    print(f"\nTraffic Flow Analysis Results:")
    print(f"- Processing time: {analysis_time:.2f} seconds")
    print(f"- Traffic flow components: {len(cliff_map.components)}")
    print(f"- Points per component: {len(data) / max(1, len(cliff_map.components)):.1f}")
    
    # Detailed component analysis
    print(f"\nTraffic Flow Components:")
    for i, comp in enumerate(cliff_map.components):
        pos = comp['position']
        direction_deg = np.degrees(comp['direction']) % 360
        flow_type = classify_traffic_flow(comp)
        
        print(f"  Flow {i+1}: {flow_type}")
        print(f"    Location: ({pos[0]:.2f}, {pos[1]:.2f})")
        print(f"    Direction: {direction_deg:.1f}°")
        print(f"    Intensity: {comp['weight']:.3f}")
    
    return cliff_map, data, metadata, analysis_time


def determine_optimal_bandwidth(data):
    """Determine optimal bandwidth based on data characteristics."""
    # Calculate inter-point distances
    if len(data) > 1000:
        # Sample for efficiency
        sample_indices = np.random.choice(len(data), 1000, replace=False)
        sample_data = data[sample_indices, :2]
    else:
        sample_data = data[:, :2]
    
    # Calculate median nearest neighbor distance
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=min(10, len(sample_data)), algorithm='auto')
    nbrs.fit(sample_data)
    distances, _ = nbrs.kneighbors(sample_data)
    
    # Use median of k-nearest neighbor distances (excluding self)
    median_nn_dist = np.median(distances[:, 1:])
    
    # Set bandwidth as fraction of median distance
    optimal_bandwidth = median_nn_dist * 0.8
    
    # Ensure reasonable bounds
    optimal_bandwidth = np.clip(optimal_bandwidth, 0.1, 2.0)
    
    return optimal_bandwidth


def classify_traffic_flow(component):
    """Classify traffic flow component based on characteristics."""
    weight = component['weight']
    direction = component['direction']
    
    # Classify by intensity
    if weight > 0.3:
        intensity = "Major"
    elif weight > 0.1:
        intensity = "Moderate"
    else:
        intensity = "Minor"
    
    # Classify by direction
    direction_deg = np.degrees(direction) % 360
    
    if direction_deg < 45 or direction_deg >= 315:
        direction_name = "Eastbound"
    elif 45 <= direction_deg < 135:
        direction_name = "Northbound"
    elif 135 <= direction_deg < 225:
        direction_name = "Westbound"
    else:
        direction_name = "Southbound"
    
    return f"{intensity} {direction_name} Traffic"


def batch_processing_example():
    """Demonstrate batch processing of multiple ATC files."""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING MULTIPLE ATC FILES")
    print("=" * 60)
    
    # Get available files
    atc_files = discover_atc_files()
    if len(atc_files) < 2:
        print("Need at least 2 ATC files for batch processing example")
        return {}
    
    # Process first few files
    files_to_process = atc_files[:3]  # Process first 3 files
    results = {}
    
    print(f"\nProcessing {len(files_to_process)} files...")
    
    for i, file_path in enumerate(files_to_process):
        print(f"\n--- Processing file {i+1}/{len(files_to_process)} ---")
        
        try:
            result = analyze_single_atc_file(file_path)
            if result is not None:
                cliff_map, data, metadata, analysis_time = result
                results[metadata['filename']] = {
                    'cliff_map': cliff_map,
                    'data': data,
                    'metadata': metadata,
                    'analysis_time': analysis_time
                }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Batch analysis summary
    print(f"\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    
    if results:
        total_points = sum(r['metadata']['processed_points'] for r in results.values())
        total_components = sum(len(r['cliff_map'].components) for r in results.values())
        total_time = sum(r['analysis_time'] for r in results.values())
        
        print(f"Successfully processed {len(results)} files:")
        print(f"- Total data points: {total_points}")
        print(f"- Total traffic components: {total_components}")
        print(f"- Total processing time: {total_time:.2f} seconds")
        print(f"- Average components per file: {total_components / len(results):.1f}")
        print(f"- Processing rate: {total_points / total_time:.0f} points/second")
        
        # Individual file summary
        print(f"\nPer-file results:")
        for filename, result in results.items():
            components = len(result['cliff_map'].components)
            points = result['metadata']['processed_points']
            time_taken = result['analysis_time']
            
            print(f"  {filename}: {components} components from {points} points ({time_taken:.1f}s)")
    
    return results


def temporal_analysis_example(results):
    """Analyze temporal patterns in traffic flow."""
    if not results:
        print("No results available for temporal analysis")
        return
    
    print("\n" + "=" * 60)
    print("TEMPORAL TRAFFIC FLOW ANALYSIS")
    print("=" * 60)
    
    # Extract temporal information from filenames
    temporal_data = []
    
    for filename, result in results.items():
        # Try to extract timestamp from filename (assuming format like atc-YYYYMMDD-HH_*.csv)
        try:
            parts = filename.replace('.csv', '').split('-')
            if len(parts) >= 3:
                date_str = parts[1]  # YYYYMMDD
                time_str = parts[2]  # HH or HHMM
                
                # Parse time
                if '_' in time_str:
                    time_str = time_str.split('_')[0]
                
                hour = int(time_str[:2]) if len(time_str) >= 2 else 0
                
                temporal_data.append({
                    'filename': filename,
                    'date': date_str,
                    'hour': hour,
                    'components': len(result['cliff_map'].components),
                    'total_weight': sum(c['weight'] for c in result['cliff_map'].components),
                    'data_points': result['metadata']['processed_points']
                })
        except:
            # Skip files with unparseable names
            continue
    
    if not temporal_data:
        print("Could not extract temporal information from filenames")
        return
    
    # Analyze patterns
    df_temporal = pd.DataFrame(temporal_data)
    
    print(f"Temporal analysis of {len(temporal_data)} time periods:")
    
    if 'hour' in df_temporal.columns:
        # Hour-based analysis
        hourly_stats = df_temporal.groupby('hour').agg({
            'components': ['mean', 'std'],
            'total_weight': ['mean', 'std'],
            'data_points': ['mean', 'std']
        }).round(2)
        
        print(f"\nHourly traffic patterns:")
        print(hourly_stats)
        
        # Find peak hours
        peak_components_hour = df_temporal.loc[df_temporal['components'].idxmax(), 'hour']
        peak_weight_hour = df_temporal.loc[df_temporal['total_weight'].idxmax(), 'hour']
        
        print(f"\nPeak activity:")
        print(f"- Most components at: {peak_components_hour:02d}:00")
        print(f"- Highest flow intensity at: {peak_weight_hour:02d}:00")
    
    # Create temporal visualization
    create_temporal_visualization(df_temporal, "atc_results")
    
    return df_temporal


def create_temporal_visualization(df_temporal, output_dir):
    """Create visualizations for temporal traffic analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'hour' in df_temporal.columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Components over time
        axes[0, 0].plot(df_temporal['hour'], df_temporal['components'], 'o-', linewidth=2, markersize=6)
        axes[0, 0].set_title('Traffic Components by Hour')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Number of Components')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Flow intensity over time
        axes[0, 1].plot(df_temporal['hour'], df_temporal['total_weight'], 's-', 
                       color='orange', linewidth=2, markersize=6)
        axes[0, 1].set_title('Traffic Flow Intensity by Hour')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Total Flow Weight')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Data points over time
        axes[1, 0].plot(df_temporal['hour'], df_temporal['data_points'], '^-', 
                       color='green', linewidth=2, markersize=6)
        axes[1, 0].set_title('Data Volume by Hour')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Number of Data Points')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Components vs data points correlation
        axes[1, 1].scatter(df_temporal['data_points'], df_temporal['components'], 
                          s=80, alpha=0.7, c=df_temporal['hour'], cmap='viridis')
        axes[1, 1].set_title('Components vs Data Volume')
        axes[1, 1].set_xlabel('Data Points')
        axes[1, 1].set_ylabel('Components')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add colorbar for hour
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label('Hour of Day')
        
        plt.tight_layout()
        
        temporal_path = os.path.join(output_dir, "traffic_temporal_analysis.png")
        plt.savefig(temporal_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Temporal analysis visualization saved to: {temporal_path}")


def create_traffic_visualizations(results):
    """Create comprehensive visualizations for traffic analysis."""
    print("\n" + "=" * 60)
    print("CREATING TRAFFIC FLOW VISUALIZATIONS")
    print("=" * 60)
    
    output_dir = "atc_results"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = FlowFieldVisualizer(figsize=(12, 8))
    
    # Visualize each processed file
    for filename, result in results.items():
        print(f"\nCreating visualizations for {filename}...")
        
        base_name = filename.replace('.csv', '')
        cliff_map = result['cliff_map']
        data = result['data']
        
        # Component visualization
        fig, ax = visualizer.plot_components(
            cliff_map,
            save_path=os.path.join(output_dir, f"{base_name}_components.png"),
            show_ellipses=True
        )
        plt.close(fig)
        
        # Flow field visualization
        fig, ax = visualizer.plot_flow_field(
            cliff_map,
            resolution=25,
            save_path=os.path.join(output_dir, f"{base_name}_flow_field.png")
        )
        plt.close(fig)
    
    # Comparative analysis
    if len(results) > 1:
        print("\nCreating comparative analysis...")
        
        # Extract CLiFF-map objects for comparison
        cliff_maps = {filename.replace('.csv', ''): result['cliff_map'] 
                     for filename, result in results.items()}
        
        fig, axes = visualizer.compare_results(
            cliff_maps,
            save_path=os.path.join(output_dir, "traffic_comparison.png")
        )
        plt.close(fig)
    
    print(f"\nAll traffic visualizations saved to: {output_dir}/")


def main():
    """Run complete ATC traffic analysis example."""
    print("CLiFF-map ATC Traffic Flow Analysis Example")
    print("===========================================")
    
    try:
        # 1. Discover available data
        atc_files = discover_atc_files()
        if not atc_files:
            print("No ATC data files found. Please ensure ATC CSV files exist in the 'atc' directory.")
            return
        
        # 2. Single file analysis demonstration
        if atc_files:
            print("\n" + "=" * 60)
            print("SINGLE FILE DEMONSTRATION")
            print("=" * 60)
            
            demo_result = analyze_single_atc_file(atc_files[0])
            if demo_result is None:
                print("Could not analyze demonstration file")
                return
        
        # 3. Batch processing
        batch_results = batch_processing_example()
        
        # 4. Temporal analysis
        temporal_df = temporal_analysis_example(batch_results)
        
        # 5. Create visualizations
        create_traffic_visualizations(batch_results)
        
        print("\n" + "=" * 60)
        print("ATC TRAFFIC ANALYSIS COMPLETE")
        print("=" * 60)
        print("- Processed multiple ATC data files")
        print("- Identified traffic flow components")
        print("- Analyzed temporal patterns")
        print("- All results saved to 'atc_results' directory")
        
        if batch_results:
            total_files = len(batch_results)
            total_components = sum(len(r['cliff_map'].components) for r in batch_results.values())
            print(f"- Summary: {total_components} traffic components from {total_files} files")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()