# CLiFF-map: Circular-Linear Flow Field Mapping

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Status](https://img.shields.io/badge/status-stable-green.svg)

A Python implementation of CLiFF-map (Circular-Linear Flow Field Mapping) for analyzing dynamic flow patterns in 2D environments using directional statistics, clustering algorithms, and probabilistic modeling.

## 🚀 Features

- **Parallel Processing**: Multi-core support for large datasets
- **Progress Monitoring**: Verbose mode with progress bars and detailed logging
- **Checkpoint/Resume**: Continue interrupted processing sessions
- **Interactive Visualization**: Complete flow field mapping and single batch analysis
- **Multiple Data Formats**: Support for air flow, pedestrian, and ATC data
- **Robust Algorithms**: Enhanced numerical stability and error handling
- **Export Capabilities**: Save results to CSV, XML, and visualization formats

## 📦 Installation

### From Source

```bash
git clone https://github.com/your-repo/cliffmap.git
cd cliffmap
pip install -e .
```

### Dependencies

```bash
pip install numpy scipy matplotlib scikit-learn pandas tqdm
```

## 🎯 Quick Start

### Basic Usage

```python
import cliffmap
from cliffmap import DynamicMap

# Load your flow data
dm = DynamicMap()
dm.load_data('your_data.csv', data_type='air_flow')

# Process with parallel computing
dm.process(n_jobs=4, verbose=True, checkpoint='my_session.pkl')

# Visualize results
dm.plot_flow_field(save_path='flow_map.png')
dm.plot_batch(batch_id=5, save_path='batch_5.png')

# Export results
dm.save_results('results.csv')
dm.save_xml('flow_field.xml')
```

### Advanced Configuration

```python
# Configure processing parameters
dm.configure(
    resolution=1.0,
    radius=0.5,
    bandwidth='auto',
    max_iterations=100,
    convergence_threshold=0.001
)

# Process with custom settings
results = dm.process(
    n_jobs=-1,              # Use all CPU cores
    verbose=True,           # Show progress
    checkpoint_interval=50, # Save every 50 batches
    silent_mode=False       # Enable logging
)
```

## 📊 Supported Data Formats

### Air Flow Data
```csv
timestamp,x,y,vx,vy,location_id
74997,0.43,0.4,-0.06,-0.03,1
75249,0.43,0.4,-0.05,-0.04,1
...
```

### Pedestrian Data
```csv
id,timestamp,x,y,vx,vy
0,1467718789217051300,49.15,5.84,-0.87,0.04
1,1467718789217051300,50.41,4.28,-1.25,-0.07
...
```

### ATC Data
```csv
timestamp,x,y,direction,speed
1635724800,100.5,200.3,1.57,5.2
1635724801,101.2,201.1,1.55,5.4
...
```

## 🔧 Configuration Options

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `resolution` | Grid resolution (meters) | 1.0 | 0.1-10.0 |
| `radius` | Clustering radius | 0.5 | 0.1-5.0 |
| `n_jobs` | Number of CPU cores | 1 | 1 to CPU count |
| `bandwidth` | Mean shift bandwidth | 'auto' | 'auto' or float |
| `max_iterations` | Maximum EM iterations | 100 | 10-1000 |
| `verbose` | Show progress | True | True/False |
| `checkpoint_interval` | Save frequency | 10 | 1-100 |

## 📈 Performance Guide

### Memory Requirements
- **Small datasets** (< 10K points): 100-500 MB
- **Medium datasets** (10K-100K points): 500MB-2GB  
- **Large datasets** (> 100K points): 2GB-8GB

### Processing Time Estimates
- **Air flow data** (3.6K points): ~1-2 minutes
- **Pedestrian data** (42K points): ~15-30 minutes
- **ATC data** (varies): Depends on temporal resolution

### Optimal Settings
```python
# For speed (less accuracy)
dm.configure(resolution=2.0, max_iterations=50)

# For accuracy (slower)
dm.configure(resolution=0.5, max_iterations=200)

# For large datasets
dm.configure(resolution=1.5, n_jobs=-1, checkpoint_interval=20)
```

## 🎨 Visualization Examples

### Flow Field Visualization
```python
# Complete flow field with all components
dm.plot_flow_field(
    colormap='hsv',
    scale_factor=3.0,
    show_confidence=True,
    save_path='complete_map.png'
)

# Filtered high-confidence flows only
dm.plot_flow_field(
    confidence_threshold=0.7,
    save_path='high_confidence_map.png'
)
```

### Single Batch Analysis
```python
# Detailed batch visualization
dm.plot_batch(
    batch_id=12,
    show_clusters=True,
    show_raw_data=True,
    show_statistics=True,
    save_path='batch_analysis.png'
)
```

### Statistical Summaries
```python
# Component distribution analysis
dm.plot_statistics(
    plot_types=['weights', 'directions', 'magnitudes'],
    save_path='statistics.png'
)
```

## 💾 Checkpoint and Resume

### Save Processing State
```python
# Automatic checkpointing during processing
dm.process(checkpoint='session.pkl', checkpoint_interval=25)

# Manual checkpoint
dm.save_checkpoint('manual_save.pkl')
```

### Resume Interrupted Session
```python
# Load and continue processing
dm.load_checkpoint('session.pkl')
dm.continue_processing(verbose=True)
```

## 📋 Examples

Comprehensive examples are provided in the `examples/` directory:

- **`air_flow_example.py`** - Air flow data processing
- **`pedestrian_example.py`** - Pedestrian movement analysis  
- **`atc_example.py`** - Air traffic control data processing
- **`parallel_processing_demo.py`** - Performance optimization
- **`visualization_gallery.py`** - Complete visualization showcase
- **`checkpoint_resume_demo.py`** - Session management example

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_algorithms.py
python -m pytest tests/test_visualization.py
python -m pytest tests/test_performance.py
```

## 📚 API Reference

### DynamicMap Class

#### Core Methods
- `load_data(filepath, data_type)` - Load flow data
- `configure(**kwargs)` - Set processing parameters  
- `process(n_jobs, verbose, checkpoint)` - Execute CLiFF-map algorithm
- `save_results(filepath)` - Export components to CSV
- `save_xml(filepath)` - Export to XML format

#### Visualization Methods
- `plot_flow_field(**kwargs)` - Complete flow field visualization
- `plot_batch(batch_id, **kwargs)` - Single batch analysis
- `plot_statistics(**kwargs)` - Statistical summaries
- `plot_input_data(**kwargs)` - Raw data visualization

#### State Management
- `save_checkpoint(filepath)` - Save current state
- `load_checkpoint(filepath)` - Restore saved state
- `continue_processing()` - Resume interrupted processing

### Batch Class

#### Analysis Methods
- `mean_shift_clustering()` - Apply Mean Shift algorithm
- `em_algorithm()` - Expectation-Maximization fitting
- `score_fit()` - Compute goodness-of-fit metrics
- `get_statistics()` - Extract batch statistics

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-repo/cliffmap.git
cd cliffmap
pip install -e ".[dev]"
pre-commit install
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Include comprehensive docstrings
- Write tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use CLiFF-map in your research, please cite:

```bibtex
@article{cliffmap2024,
  
}
```

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/tkucner/cliffmap/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tkucner/cliffmap/discussions)

## 🔬 Research Applications

CLiFF-map has been successfully applied to:

- **Environmental Flow Analysis** - Wind patterns, ocean currents
- **Urban Planning** - Pedestrian traffic optimization  
- **Robotics** - Navigation in dynamic environments
- **Transportation** - Traffic flow modeling and prediction
- **Biology** - Animal movement pattern analysis

## 🏆 Acknowledgments

- Original MATLAB implementation by Tomasz Kucner
- NumPy and SciPy communities for numerical computing tools
- Matplotlib developers for visualization capabilities
- Contributors and users of the CLiFF-map project

---

**Made with ❤️ for the scientific computing community**
