#!/usr/bin/env python3
"""
Test script for enhanced DynamicMap with automatic column detection
and support for directional/velocity data formats.
"""

import numpy as np
import pandas as pd
import os
import sys

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cliffmap import DynamicMap

def create_test_data():
    """Create test datasets in different formats."""
    np.random.seed(42)
    n_points = 200
    
    # Test data 1: Directional format (x, y, direction, speed)
    print("Creating directional format test data...")
    x = np.random.uniform(0, 10, n_points)
    y = np.random.uniform(0, 10, n_points)
    direction = np.random.uniform(0, 2*np.pi, n_points)
    speed = np.random.lognormal(0, 0.3, n_points)
    
    df_directional = pd.DataFrame({
        'x_position': x,
        'y_position': y,
        'heading': direction,
        'velocity': speed
    })
    df_directional.to_csv('test_directional.csv', index=False)
    
    # Test data 2: Velocity format (x, y, vx, vy)
    print("Creating velocity format test data...")
    vx = speed * np.cos(direction)
    vy = speed * np.sin(direction)
    
    df_velocity = pd.DataFrame({
        'pos_x': x,
        'pos_y': y,
        'vel_x': vx,
        'vel_y': vy
    })
    df_velocity.to_csv('test_velocity.csv', index=False)
    
    # Test data 3: Standard column names
    print("Creating standard format test data...")
    df_standard = pd.DataFrame({
        'x': x,
        'y': y,
        'direction': direction,
        'speed': speed
    })
    df_standard.to_csv('test_standard.csv', index=False)
    
    return ['test_directional.csv', 'test_velocity.csv', 'test_standard.csv']

def test_automatic_detection():
    """Test automatic column detection."""
    print("\n" + "="*60)
    print("TESTING AUTOMATIC COLUMN DETECTION")
    print("="*60)
    
    files = create_test_data()
    
    for filename in files:
        print(f"\nTesting file: {filename}")
        print("-" * 40)
        
        # Test automatic detection
        cliff_map = DynamicMap(verbose=True)
        cliff_map.load_data(filename)
        
        print(f"Detected format: {cliff_map.data_format}")
        print(f"Column mapping: {cliff_map.column_info['mapping']}")
        print(f"Data shape: {cliff_map.data.shape}")
        
        # Quick fit test
        cliff_map.batch_size = 50
        cliff_map.max_iterations = 20
        cliff_map.fit()
        
        print(f"Found {len(cliff_map.components)} components")
        
        # Test XML save
        xml_filename = filename.replace('.csv', '_result.xml')
        cliff_map.save_xml(xml_filename)
        print(f"Saved XML: {xml_filename}")
        
        # Test CSV save
        csv_filename = filename.replace('.csv', '_components.csv')
        cliff_map.save_csv(csv_filename)
        print(f"Saved CSV: {csv_filename}")

def test_custom_mapping():
    """Test custom column mapping."""
    print("\n" + "="*60)
    print("TESTING CUSTOM COLUMN MAPPING")
    print("="*60)
    
    # Create data with unusual column names
    np.random.seed(42)
    n_points = 100
    
    df = pd.DataFrame({
        'longitude': np.random.uniform(-1, 1, n_points),
        'latitude': np.random.uniform(-1, 1, n_points),
        'bearing': np.random.uniform(0, 360, n_points),  # degrees
        'wind_speed': np.random.lognormal(0, 0.2, n_points)
    })
    df.to_csv('test_custom.csv', index=False)
    
    # Test with custom mapping
    custom_mapping = {
        'x': 'longitude',
        'y': 'latitude', 
        'direction': 'bearing',
        'speed': 'wind_speed'
    }
    
    print(f"Using custom mapping: {custom_mapping}")
    
    cliff_map = DynamicMap(
        column_mapping=custom_mapping,
        verbose=True,
        batch_size=30,
        max_iterations=15
    )
    
    cliff_map.load_data('test_custom.csv')
    cliff_map.fit()
    
    print(f"Found {len(cliff_map.components)} components")
    
    # Test XML save with custom data
    cliff_map.save_xml('test_custom_result.xml', include_metadata=True)
    print("Saved custom mapping result to XML")
    
    # Display component summary
    summary = cliff_map.get_component_summary()
    print(f"\nComponent Summary:")
    print(f"Total components: {summary['n_components']}")
    print(f"Total weight: {summary['total_weight']:.3f}")
    
    for comp in summary['components'][:3]:  # Show first 3
        print(f"  Component {comp['id']}: {comp['classification']['flow_type']}")

def test_velocity_components():
    """Test velocity component data format."""
    print("\n" + "="*60)
    print("TESTING VELOCITY COMPONENT FORMAT")
    print("="*60)
    
    # Create velocity component data
    np.random.seed(42)
    
    # Create circular flow pattern
    n_points = 150
    r = np.random.uniform(1, 3, n_points)
    theta = np.random.uniform(0, 2*np.pi, n_points)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Tangential velocity (circular flow)
    vx = -r * 0.5 * np.sin(theta) + np.random.normal(0, 0.1, n_points)
    vy = r * 0.5 * np.cos(theta) + np.random.normal(0, 0.1, n_points)
    
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'u_component': vx,
        'v_component': vy
    })
    df.to_csv('test_circular_flow.csv', index=False)
    
    # Test with automatic detection
    cliff_map = DynamicMap(verbose=True)
    cliff_map.load_data('test_circular_flow.csv')
    
    print(f"Detected format: {cliff_map.data_format}")
    print(f"Original velocity data converted to directional format")
    
    cliff_map.fit()
    
    print(f"Found {len(cliff_map.components)} components in circular flow")
    
    # Save results
    cliff_map.save_xml('circular_flow_result.xml')
    cliff_map.save_csv('circular_flow_components.csv', include_velocity_components=True)
    
    print("Circular flow analysis complete")

def main():
    """Run all tests."""
    print("CLiFF-map Enhanced Data Loading Test Suite")
    print("==========================================")
    
    try:
        # Test 1: Automatic detection
        test_automatic_detection()
        
        # Test 2: Custom mapping
        test_custom_mapping()
        
        # Test 3: Velocity components
        test_velocity_components()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nGenerated files:")
        for f in os.listdir('.'):
            if f.endswith(('.csv', '.xml')) and f.startswith('test'):
                print(f"  - {f}")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test files
        print(f"\nCleaning up test files...")
        for f in os.listdir('.'):
            if f.startswith('test_') and f.endswith(('.csv', '.xml')):
                try:
                    os.remove(f)
                    print(f"  Removed {f}")
                except:
                    pass
        
        for f in ['circular_flow_result.xml', 'circular_flow_components.csv', 'test_custom_result.xml']:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"  Removed {f}")
            except:
                pass

if __name__ == "__main__":
    main()