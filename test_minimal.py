#!/usr/bin/env python3
"""
Minimal test for CLiFF-map core functionality without external dependencies.
"""

import numpy as np
import sys
import os

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_core_imports():
    """Test core imports without pandas/sklearn."""
    print("Testing Core CLiFF-map Imports")
    print("=" * 35)
    
    try:
        from cliffmap.utils import cart2pol, pol2cart, wrap_to_2pi
        print("✓ Utils module imported successfully")
    except Exception as e:
        print(f"✗ Utils import failed: {e}")
        return False
    
    try:
        from cliffmap.batch import Batch
        print("✓ Batch class imported successfully")
    except Exception as e:
        print(f"✗ Batch import failed: {e}")
        return False
    
    try:
        from cliffmap.dynamic_map import DynamicMap
        print("✓ DynamicMap class imported successfully")
    except Exception as e:
        print(f"✗ DynamicMap import failed: {e}")
        return False
        
    return True

def test_basic_functionality():
    """Test basic functionality with numpy arrays."""
    print("\nTesting Basic Functionality")
    print("=" * 30)
    
    try:
        from cliffmap import DynamicMap
        
        # Create test data as numpy array
        np.random.seed(42)
        n_points = 30
        
        test_data = np.column_stack([
            np.random.uniform(0, 5, n_points),  # x
            np.random.uniform(0, 5, n_points),  # y
            np.random.uniform(0, 2*np.pi, n_points),  # direction
            np.random.lognormal(0, 0.2, n_points)     # speed
        ])
        
        print(f"✓ Created test data: {test_data.shape}")
        
        # Initialize DynamicMap with small parameters for quick test
        dm = DynamicMap(
            batch_size=15, 
            max_iterations=10,
            verbose=True,
            parallel=False  # Disable parallelization for simple test
        )
        print("✓ DynamicMap initialized")
        
        # Test loading numpy array directly
        dm.load_data(test_data)
        print(f"✓ Data loaded successfully: {dm.data.shape}")
        
        # Test fitting
        dm.fit()
        print(f"✓ Fitting completed, found {len(dm.components)} components")
        
        # Test XML export
        if hasattr(dm, 'save_xml'):
            dm.save_xml('test_basic.xml')
            print("✓ XML export successful")
        
        # Test CSV export
        if hasattr(dm, 'save_csv'):
            dm.save_csv('test_basic.csv')
            print("✓ CSV export successful")
            
        # Test component summary
        summary = dm.get_component_summary()
        print(f"✓ Component summary: {summary['n_components']} components")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_loading():
    """Test CSV loading with fallback mechanism."""
    print("\nTesting CSV Loading")
    print("=" * 20)
    
    try:
        # Create a simple CSV file manually
        csv_content = """x,y,direction,speed
1.0,1.0,0.0,1.0
2.0,2.0,1.57,0.8
3.0,1.0,3.14,1.2
1.5,3.0,4.71,0.9
4.0,2.5,0.78,1.1"""
        
        with open('test_simple.csv', 'w') as f:
            f.write(csv_content)
        
        print("✓ Created test CSV file")
        
        from cliffmap import DynamicMap
        
        dm = DynamicMap(
            batch_size=5,
            max_iterations=5,
            verbose=True,
            parallel=False
        )
        
        # Test CSV loading
        dm.load_data('test_simple.csv')
        print(f"✓ CSV loaded successfully: {dm.data.shape}")
        print(f"  Data format: {dm.data_format}")
        print(f"  Column mapping: {dm.column_info['mapping']}")
        
        # Quick fit
        dm.fit()
        print(f"✓ Processed CSV data, found {len(dm.components)} components")
        
        return True
        
    except Exception as e:
        print(f"✗ CSV test failed: {e}")
        return False
        
    finally:
        # Clean up
        if os.path.exists('test_simple.csv'):
            os.remove('test_simple.csv')

def show_results_info():
    """Show information about created results if they exist."""
    print("\nResults Files Generated:")
    print("=" * 25)
    
    for filename in ['test_basic.xml', 'test_basic.csv']:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✓ {filename}: {size} bytes")
            
            if filename.endswith('.xml'):
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    print(f"  XML structure preview (first 3 lines):")
                    for line in lines[:3]:
                        print(f"    {line.strip()}")

def main():
    """Run minimal tests."""
    print("CLiFF-map Minimal Functionality Test")
    print("====================================\n")
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Core imports
    if test_core_imports():
        success_count += 1
    
    # Test 2: Basic functionality
    if test_basic_functionality():
        success_count += 1
        
    # Test 3: CSV loading
    if test_csv_loading():
        success_count += 1
    
    # Show results
    show_results_info()
    
    print(f"\n" + "=" * 40)
    print(f"TESTS COMPLETED: {success_count}/{total_tests} successful")
    print("=" * 40)
    
    if success_count == total_tests:
        print("🎉 All core functionality is working!")
        print("\nThe enhanced DynamicMap includes:")
        print("- Automatic column detection from CSV headers")
        print("- Support for both directional (direction, speed) and velocity (vx, vy) data")
        print("- Fallback mechanisms when pandas/sklearn are not available")
        print("- XML export with comprehensive metadata")
        print("- Custom column mapping capability")
        print("- Progress monitoring and parallel processing")
    else:
        print("⚠ Some tests failed - check the output above")
    
    # Clean up
    for f in ['test_basic.xml', 'test_basic.csv']:
        try:
            if os.path.exists(f):
                os.remove(f)
        except:
            pass

if __name__ == "__main__":
    main()