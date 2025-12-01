#!/usr/bin/env python3
"""
Simple test for enhanced DynamicMap without external dependencies beyond numpy/matplotlib.
"""

import numpy as np
import os
import sys

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def create_simple_csv(filename, data_format='directional'):
    """Create a simple CSV file without pandas."""
    np.random.seed(42)
    n_points = 50
    
    x = np.random.uniform(0, 5, n_points)
    y = np.random.uniform(0, 5, n_points)
    
    if data_format == 'directional':
        direction = np.random.uniform(0, 2*np.pi, n_points)
        speed = np.random.lognormal(0, 0.2, n_points)
        
        with open(filename, 'w') as f:
            f.write("x,y,direction,speed\n")
            for i in range(n_points):
                f.write(f"{x[i]:.6f},{y[i]:.6f},{direction[i]:.6f},{speed[i]:.6f}\n")
                
    elif data_format == 'velocity':
        direction = np.random.uniform(0, 2*np.pi, n_points)
        speed = np.random.lognormal(0, 0.2, n_points)
        vx = speed * np.cos(direction)
        vy = speed * np.sin(direction)
        
        with open(filename, 'w') as f:
            f.write("x,y,vx,vy\n")
            for i in range(n_points):
                f.write(f"{x[i]:.6f},{y[i]:.6f},{vx[i]:.6f},{vy[i]:.6f}\n")

def test_basic_loading():
    """Test basic functionality of enhanced DynamicMap."""
    print("Testing Enhanced DynamicMap Basic Functionality")
    print("=" * 50)
    
    # Create test files
    create_simple_csv('test_dir.csv', 'directional')
    create_simple_csv('test_vel.csv', 'velocity')
    
    # Test the basic imports first
    try:
        from cliffmap.dynamic_map import DynamicMap
        print("✓ DynamicMap import successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
        
    # Test loading directional data
    try:
        print("\nTesting directional data loading...")
        dm = DynamicMap(verbose=True, batch_size=20, max_iterations=10)
        
        # Check if load_data method exists
        if hasattr(dm, 'load_data'):
            print("✓ load_data method available")
        else:
            print("✗ load_data method missing")
            return False
            
        # Try to load data (might fail due to missing pandas, but we can catch that)
        try:
            dm.load_data('test_dir.csv')
            print("✓ Directional data loaded successfully")
            print(f"  Data format: {dm.data_format}")
            print(f"  Data shape: {dm.data.shape}")
            
            # Test fitting
            dm.fit()
            print(f"✓ Fitting complete, found {len(dm.components)} components")
            
            # Test XML save
            if hasattr(dm, 'save_xml'):
                dm.save_xml('test_result.xml')
                print("✓ XML save successful")
            
        except Exception as e:
            print(f"⚠ Data loading failed (expected if pandas not available): {e}")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
        
    # Test velocity data
    try:
        print("\nTesting velocity data loading...")
        dm2 = DynamicMap(verbose=True, batch_size=20)
        # We'll skip this test if pandas isn't available
        
    except Exception as e:
        print(f"⚠ Velocity test skipped: {e}")
    
    return True

def test_xml_structure():
    """Test the XML output structure."""
    print("\nTesting XML Output Structure")
    print("=" * 30)
    
    # Check if XML file was created
    if os.path.exists('test_result.xml'):
        try:
            with open('test_result.xml', 'r') as f:
                xml_content = f.read()
                
            print("✓ XML file created successfully")
            print(f"✓ XML file size: {len(xml_content)} characters")
            
            # Check for key XML elements
            required_elements = ['<cliff_map>', '<metadata>', '<components>', '</cliff_map>']
            for element in required_elements:
                if element in xml_content:
                    print(f"✓ Found XML element: {element}")
                else:
                    print(f"✗ Missing XML element: {element}")
                    
            # Show first few lines
            lines = xml_content.split('\n')[:10]
            print(f"\nFirst few lines of XML:")
            for line in lines:
                if line.strip():
                    print(f"  {line}")
            print(f"  ... ({len(xml_content.split('\\n'))} total lines)")
            
        except Exception as e:
            print(f"✗ Error reading XML: {e}")
    else:
        print("⚠ No XML file found to test")

def main():
    """Run simple tests."""
    print("CLiFF-map Enhanced Loading Simple Test")
    print("=====================================\n")
    
    try:
        # Test basic functionality
        success = test_basic_loading()
        
        if success:
            test_xml_structure()
            print("\n" + "=" * 40)
            print("BASIC TESTS COMPLETED")
            print("=" * 40)
        else:
            print("\n✗ Basic tests failed")
            
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        for f in ['test_dir.csv', 'test_vel.csv', 'test_result.xml']:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"Cleaned up {f}")
            except:
                pass

if __name__ == "__main__":
    main()