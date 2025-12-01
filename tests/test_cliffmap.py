"""
Comprehensive test suite for CLiFF-map Python package.

This module provides unit tests and integration tests for all package components
including DynamicMap, Batch, visualization, and checkpointing functionality.
"""

import unittest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys

# Add package to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cliffmap import DynamicMap, Batch, FlowFieldVisualizer, CheckpointManager
from cliffmap.utils import WangDivergence, circular_mean
import pandas as pd


class TestBatch(unittest.TestCase):
    """Test cases for Batch class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_data = np.random.randn(100, 4)
        self.batch = Batch(self.test_data)
    
    def test_initialization(self):
        """Test batch initialization."""
        self.assertEqual(len(self.batch.data), 100)
        self.assertEqual(self.batch.data.shape[1], 4)
        self.assertIsNotNone(self.batch.position)
        self.assertIsNotNone(self.batch.direction)
    
    def test_mean_shift(self):
        """Test mean shift clustering."""
        components = self.batch.MeanShift2Dv(bandwidth=0.5, min_samples=5)
        
        self.assertIsInstance(components, list)
        self.assertGreater(len(components), 0)
        
        # Check component structure
        for comp in components:
            self.assertIn('position', comp)
            self.assertIn('direction', comp)
            self.assertIn('weight', comp)
            self.assertEqual(len(comp['position']), 2)
    
    def test_em_algorithm(self):
        """Test Expectation-Maximization algorithm."""
        initial_components = [
            {'position': np.array([0, 0]), 'direction': 0, 'weight': 0.5},
            {'position': np.array([1, 1]), 'direction': np.pi/2, 'weight': 0.5}
        ]
        
        refined_components = self.batch.EMv(initial_components, max_iterations=10)
        
        self.assertIsInstance(refined_components, list)
        self.assertEqual(len(refined_components), len(initial_components))
        
        # Check that weights sum to 1
        total_weight = sum(comp['weight'] for comp in refined_components)
        self.assertAlmostEqual(total_weight, 1.0, places=5)
    
    def test_plotting_functions(self):
        """Test plotting functionality."""
        # These should not raise exceptions
        try:
            self.batch.PlotMS_jitter()
            self.batch.PlotMS()
            # Note: In actual testing, you might want to mock matplotlib
        except Exception as e:
            self.fail(f"Plotting functions should not raise exceptions: {e}")


class TestDynamicMap(unittest.TestCase):
    """Test cases for DynamicMap class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Create synthetic flow data
        self.test_data = self._generate_test_flow_data()
        self.dynamic_map = DynamicMap(
            batch_size=50,
            max_iterations=10,  # Shorter for testing
            bandwidth=0.5,
            min_samples=3,
            parallel=False,  # Disable for deterministic testing
            verbose=False,
            progress=False
        )
    
    def _generate_test_flow_data(self, n_points=200):
        """Generate synthetic flow data for testing."""
        # Create two main flow components
        component1 = np.random.multivariate_normal([2, 2], [[0.1, 0], [0, 0.1]], n_points//2)
        component2 = np.random.multivariate_normal([8, 8], [[0.1, 0], [0, 0.1]], n_points//2)
        
        # Add directions
        directions1 = np.random.normal(0, 0.1, n_points//2)  # Eastward flow
        directions2 = np.random.normal(np.pi, 0.1, n_points//2)  # Westward flow
        
        # Add speeds
        speeds1 = np.random.normal(1.0, 0.1, n_points//2)
        speeds2 = np.random.normal(1.2, 0.1, n_points//2)
        
        # Combine data
        positions = np.vstack([component1, component2])
        directions = np.hstack([directions1, directions2])
        speeds = np.hstack([speeds1, speeds2])
        
        data = np.column_stack([positions, directions, speeds])
        return data
    
    def test_initialization(self):
        """Test DynamicMap initialization."""
        self.assertEqual(self.dynamic_map.batch_size, 50)
        self.assertEqual(self.dynamic_map.max_iterations, 10)
        self.assertEqual(self.dynamic_map.bandwidth, 0.5)
        self.assertEqual(self.dynamic_map.min_samples, 3)
        self.assertFalse(self.dynamic_map.parallel)
    
    def test_data_loading(self):
        """Test different data loading methods."""
        # Test loading from array
        self.dynamic_map.load_data(self.test_data)
        self.assertIsNotNone(self.dynamic_map.data)
        self.assertEqual(len(self.dynamic_map.data), len(self.test_data))
        
        # Test loading from file (create temporary file)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(self.test_data)
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            self.dynamic_map.load_data(temp_file)
            self.assertIsNotNone(self.dynamic_map.data)
        finally:
            os.unlink(temp_file)
    
    def test_fitting(self):
        """Test the main fitting process."""
        self.dynamic_map.fit(self.test_data)
        
        # Check results
        self.assertIsNotNone(self.dynamic_map.components)
        self.assertGreater(len(self.dynamic_map.components), 0)
        
        # Check component structure
        for comp in self.dynamic_map.components:
            self.assertIn('position', comp)
            self.assertIn('direction', comp)
            self.assertIn('weight', comp)
            self.assertGreater(comp['weight'], 0)
            self.assertLessEqual(comp['weight'], 1.0)
    
    def test_parallel_processing(self):
        """Test parallel processing functionality."""
        # Test with parallel enabled
        parallel_map = DynamicMap(
            batch_size=30,
            max_iterations=5,
            parallel=True,
            n_jobs=2,
            verbose=False,
            progress=False
        )
        
        parallel_map.fit(self.test_data)
        
        # Should still produce valid results
        self.assertIsNotNone(parallel_map.components)
        self.assertGreater(len(parallel_map.components), 0)
    
    def test_progress_monitoring(self):
        """Test progress monitoring functionality."""
        progress_map = DynamicMap(
            batch_size=30,
            max_iterations=5,
            progress=True,
            verbose=False
        )
        
        # Should not raise exceptions
        progress_map.fit(self.test_data)
        self.assertIsNotNone(progress_map.components)
    
    def test_history_tracking(self):
        """Test training history tracking."""
        self.dynamic_map.fit(self.test_data)
        
        if hasattr(self.dynamic_map, 'history'):
            history = self.dynamic_map.history
            self.assertIsInstance(history, dict)


class TestVisualization(unittest.TestCase):
    """Test cases for visualization components."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_data = np.random.randn(50, 4)
        
        # Create a simple DynamicMap with known components
        self.dynamic_map = DynamicMap(verbose=False, progress=False)
        self.dynamic_map.components = [
            {'position': np.array([1, 1]), 'direction': 0, 'weight': 0.6, 'uncertainty': 0.1},
            {'position': np.array([3, 3]), 'direction': np.pi/2, 'weight': 0.4, 'uncertainty': 0.2}
        ]
        self.dynamic_map.data = self.test_data
        
        self.visualizer = FlowFieldVisualizer(figsize=(8, 6))
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_components(self, mock_show):
        """Test component plotting."""
        save_path = os.path.join(self.temp_dir, "test_components.png")
        
        fig, ax = self.visualizer.plot_components(
            self.dynamic_map,
            save_path=save_path,
            show_ellipses=True
        )
        
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertTrue(os.path.exists(save_path))
    
    @patch('matplotlib.pyplot.show')
    def test_plot_flow_field(self, mock_show):
        """Test flow field plotting."""
        save_path = os.path.join(self.temp_dir, "test_flow_field.png")
        
        fig, ax = self.visualizer.plot_flow_field(
            self.dynamic_map,
            resolution=10,  # Low resolution for speed
            save_path=save_path
        )
        
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertTrue(os.path.exists(save_path))
    
    @patch('matplotlib.pyplot.show')
    def test_compare_results(self, mock_show):
        """Test results comparison plotting."""
        save_path = os.path.join(self.temp_dir, "test_comparison.png")
        
        # Create second DynamicMap for comparison
        dynamic_map2 = DynamicMap(verbose=False, progress=False)
        dynamic_map2.components = [
            {'position': np.array([2, 2]), 'direction': np.pi, 'weight': 0.5, 'uncertainty': 0.15}
        ]
        dynamic_map2.data = self.test_data
        
        results_dict = {
            'Method 1': self.dynamic_map,
            'Method 2': dynamic_map2
        }
        
        fig, axes = self.visualizer.compare_results(
            results_dict,
            save_path=save_path
        )
        
        self.assertIsNotNone(fig)
        self.assertIsNotNone(axes)
        self.assertTrue(os.path.exists(save_path))


class TestCheckpointing(unittest.TestCase):
    """Test cases for checkpointing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_data = np.random.randn(50, 4)
        
        # Create temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(self.temp_dir)
        
        # Create a fitted DynamicMap
        self.dynamic_map = DynamicMap(
            batch_size=30,
            max_iterations=5,
            verbose=False,
            progress=False
        )
        self.dynamic_map.fit(self.test_data)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        metadata = {'test': 'data', 'iteration': 1}
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.dynamic_map,
            checkpoint_name="test_checkpoint",
            metadata=metadata
        )
        
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Check summary file
        summary_path = checkpoint_path.replace('.pkl', '_summary.json')
        self.assertTrue(os.path.exists(summary_path))
    
    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        metadata = {'test': 'data', 'iteration': 1}
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.dynamic_map,
            checkpoint_name="test_checkpoint",
            metadata=metadata
        )
        
        # Load checkpoint
        loaded_map, loaded_metadata = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        self.assertIsNotNone(loaded_map)
        self.assertEqual(loaded_metadata, metadata)
        self.assertEqual(len(loaded_map.components), len(self.dynamic_map.components))
    
    def test_list_checkpoints(self):
        """Test checkpoint listing."""
        # Save multiple checkpoints
        self.checkpoint_manager.save_checkpoint(self.dynamic_map, "checkpoint1")
        self.checkpoint_manager.save_checkpoint(self.dynamic_map, "checkpoint2")
        
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        self.assertEqual(len(checkpoints), 2)
        self.assertTrue(any("checkpoint1" in cp for cp in checkpoints))
        self.assertTrue(any("checkpoint2" in cp for cp in checkpoints))
    
    def test_get_checkpoint_info(self):
        """Test checkpoint information retrieval."""
        metadata = {'test_info': 'value'}
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.dynamic_map,
            checkpoint_name="info_test",
            metadata=metadata
        )
        
        info = self.checkpoint_manager.get_checkpoint_info(checkpoint_path)
        
        self.assertIn('timestamp', info)
        self.assertIn('n_components', info)
        self.assertIn('metadata', info)
        self.assertEqual(info['metadata'], metadata)
        self.assertEqual(info['n_components'], len(self.dynamic_map.components))
    
    def test_cleanup_old_checkpoints(self):
        """Test checkpoint cleanup functionality."""
        # Create multiple checkpoints
        for i in range(7):
            self.checkpoint_manager.save_checkpoint(self.dynamic_map, f"checkpoint_{i}")
        
        initial_count = len(self.checkpoint_manager.list_checkpoints())
        self.assertEqual(initial_count, 7)
        
        # Cleanup keeping only 3
        self.checkpoint_manager.cleanup_old_checkpoints(keep_latest=3)
        
        final_count = len(self.checkpoint_manager.list_checkpoints())
        self.assertEqual(final_count, 3)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_wang_divergence(self):
        """Test Wang divergence calculation."""
        # Create test distributions
        dist1 = np.array([0.5, 0.3, 0.2])
        dist2 = np.array([0.4, 0.4, 0.2])
        
        divergence = WangDivergence(dist1, dist2)
        
        self.assertIsInstance(divergence, float)
        self.assertGreaterEqual(divergence, 0)
        
        # Test symmetry
        divergence_reverse = WangDivergence(dist2, dist1)
        self.assertAlmostEqual(divergence, divergence_reverse, places=5)
        
        # Test identity
        identity_divergence = WangDivergence(dist1, dist1)
        self.assertAlmostEqual(identity_divergence, 0, places=5)
    
    def test_circular_mean(self):
        """Test circular mean calculation."""
        # Test known values
        angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        weights = np.array([1, 1, 1, 1])
        
        mean_angle = circular_mean(angles, weights)
        
        self.assertIsInstance(mean_angle, float)
        self.assertGreaterEqual(mean_angle, 0)
        self.assertLess(mean_angle, 2*np.pi)
        
        # Test with unequal weights
        weights_unequal = np.array([2, 1, 1, 1])
        mean_weighted = circular_mean(angles, weights_unequal)
        
        self.assertNotEqual(mean_angle, mean_weighted)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        np.random.seed(42)
        # Create more realistic test data
        self.test_data = self._generate_realistic_flow_data()
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _generate_realistic_flow_data(self, n_points=300):
        """Generate realistic flow data for integration testing."""
        # Create three flow components with realistic properties
        np.random.seed(42)
        
        # Main corridor flow (eastward)
        main_flow = np.random.multivariate_normal([5, 5], [[0.2, 0], [0, 0.5]], n_points//3)
        main_directions = np.random.normal(0, 0.2, n_points//3)
        main_speeds = np.random.normal(1.5, 0.3, n_points//3)
        
        # Cross flow (northward)
        cross_flow = np.random.multivariate_normal([5, 2], [[0.5, 0], [0, 0.2]], n_points//3)
        cross_directions = np.random.normal(np.pi/2, 0.3, n_points//3)
        cross_speeds = np.random.normal(1.0, 0.2, n_points//3)
        
        # Local circulation
        local_flow = np.random.multivariate_normal([8, 8], [[0.3, 0.1], [0.1, 0.3]], n_points//3)
        local_directions = np.random.uniform(0, 2*np.pi, n_points//3)
        local_speeds = np.random.normal(0.8, 0.3, n_points//3)
        
        # Combine all data
        positions = np.vstack([main_flow, cross_flow, local_flow])
        directions = np.hstack([main_directions, cross_directions, local_directions])
        speeds = np.hstack([main_speeds, cross_speeds, local_speeds])
        
        return np.column_stack([positions, directions, speeds])
    
    def test_complete_analysis_workflow(self):
        """Test complete analysis workflow from data to results."""
        # 1. Initialize DynamicMap
        dynamic_map = DynamicMap(
            batch_size=60,
            max_iterations=20,
            bandwidth=0.6,
            min_samples=8,
            parallel=True,
            n_jobs=2,
            verbose=False,
            progress=False
        )
        
        # 2. Fit model
        dynamic_map.fit(self.test_data)
        
        # 3. Verify results
        self.assertIsNotNone(dynamic_map.components)
        self.assertGreater(len(dynamic_map.components), 0)
        self.assertLessEqual(len(dynamic_map.components), 10)  # Reasonable upper bound
        
        # 4. Test visualization
        visualizer = FlowFieldVisualizer()
        
        component_plot_path = os.path.join(self.temp_dir, "integration_components.png")
        with patch('matplotlib.pyplot.show'):
            fig, ax = visualizer.plot_components(
                dynamic_map,
                save_path=component_plot_path
            )
        
        self.assertTrue(os.path.exists(component_plot_path))
        
        # 5. Test checkpointing
        checkpoint_manager = CheckpointManager(os.path.join(self.temp_dir, "checkpoints"))
        
        checkpoint_path = checkpoint_manager.save_checkpoint(
            dynamic_map,
            checkpoint_name="integration_test",
            metadata={'test_type': 'integration', 'data_points': len(self.test_data)}
        )
        
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # 6. Test checkpoint loading
        loaded_map, metadata = checkpoint_manager.load_checkpoint(checkpoint_path)
        
        self.assertEqual(len(loaded_map.components), len(dynamic_map.components))
        self.assertEqual(metadata['test_type'], 'integration')
        self.assertEqual(metadata['data_points'], len(self.test_data))
    
    def test_performance_characteristics(self):
        """Test performance characteristics of the implementation."""
        import time
        
        # Test with different data sizes
        sizes = [100, 500, 1000]
        processing_times = []
        
        for size in sizes:
            data = self.test_data[:size]
            
            dynamic_map = DynamicMap(
                batch_size=50,
                max_iterations=10,
                verbose=False,
                progress=False,
                parallel=False  # For consistent timing
            )
            
            start_time = time.time()
            dynamic_map.fit(data)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            
            # Verify results are reasonable
            self.assertIsNotNone(dynamic_map.components)
            self.assertGreater(len(dynamic_map.components), 0)
        
        # Check that processing time scales reasonably
        # (Should not grow exponentially)
        self.assertLess(processing_times[-1] / processing_times[0], 50)  # Very generous bound


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBatch,
        TestDynamicMap,
        TestVisualization,
        TestCheckpointing,
        TestUtils,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == "__main__":
    result = run_tests()
    
    # Exit with error code if tests failed
    if not result.wasSuccessful():
        sys.exit(1)