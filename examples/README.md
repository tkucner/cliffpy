# CLiFF-map Examples

This directory contains comprehensive examples demonstrating all features of the CLiFF-map Python package.

## Available Examples

### 1. `complete_demo.py` - Complete Package Demonstration
The most comprehensive example showcasing all package features:
- ✅ Multi-dataset analysis (synthetic + real data)
- ✅ Parallel processing and performance optimization
- ✅ Progress monitoring with tqdm
- ✅ Advanced visualization and plotting
- ✅ Checkpointing and state management
- ✅ Result export and comparison
- ✅ Performance benchmarking

**Usage:**
```bash
cd examples
python complete_demo.py
```

### 2. `air_flow_example.py` - Air Flow Analysis
Focused example using air flow sensor data:
- Basic CLiFF-map analysis workflow
- Visualization of flow components and fields
- Checkpointing demonstration
- Parameter comparison
- Result export to CSV/XML

**Usage:**
```bash
python air_flow_example.py
```

### 3. `pedestrian_flow_example.py` - Pedestrian Flow Analysis
Advanced example for large pedestrian datasets:
- Large dataset handling and optimization
- Parallel processing benchmarks
- Interrupted training with checkpointing
- Advanced flow visualizations
- Temporal pattern analysis

**Usage:**
```bash
python pedestrian_flow_example.py
```

### 4. `atc_traffic_example.py` - Traffic Counter Analysis
Traffic data analysis with temporal features:
- Batch processing of multiple files
- Temporal pattern analysis
- Traffic flow classification
- Automated bandwidth selection
- Multi-file comparison and visualization

**Usage:**
```bash
python atc_traffic_example.py
```

## Data Requirements

The examples are designed to work with the following data files (place in project root):

### Required Files:
- `air_flow.csv` - Air flow sensor data (x, y, direction, speed)
- `pedestrian.csv` - Pedestrian movement data
- `atc/` directory - ATC traffic counter CSV files

### Data Format:
All CSV files should contain at least:
- Column 1: X position
- Column 2: Y position  
- Column 3: Direction (radians) [optional, will be computed if missing]
- Column 4: Speed/magnitude [optional, defaults to 1.0]

## Output Structure

Each example creates organized output directories:

```
example_results/
├── dataset_analysis/
│   ├── components.png          # Flow component visualization
│   ├── flow_field.png         # Interpolated flow field
│   ├── training_history.png   # Training convergence
│   ├── data_distribution.png  # Input data analysis
│   ├── components.csv         # Exported component data
│   └── checkpoints/          # Saved model states
├── comparison.png            # Multi-method comparison
└── analysis_summary.csv     # Performance metrics
```

## Features Demonstrated

### Core Analysis:
- CLiFF-map flow field fitting
- Mean Shift clustering for initial components
- Expectation-Maximization refinement
- Component weight and direction estimation

### Advanced Features:
- **Parallel Processing**: ThreadPoolExecutor-based parallelization
- **Progress Monitoring**: tqdm progress bars for long-running analysis
- **Checkpointing**: Save/restore analysis state for interrupted training
- **Adaptive Parameters**: Automatic bandwidth and parameter selection
- **Memory Efficiency**: Batch processing for large datasets

### Visualization:
- Flow component plots with uncertainty ellipses
- Interpolated flow field visualizations
- Training convergence and history plots
- Data distribution analysis
- Multi-dataset comparison plots
- Flow intensity heatmaps

### Export and Integration:
- CSV export of component parameters
- XML export for GIS integration
- JSON summaries with metadata
- Checkpoint files for resumable analysis
- Performance benchmarking reports

## Performance Tips

### For Large Datasets:
```python
# Optimize for datasets > 10,000 points
cliff_map = DynamicMap(
    batch_size=200,          # Larger batches for efficiency
    parallel=True,           # Enable parallel processing
    n_jobs=4,               # Use multiple cores
    min_samples=20,         # Higher threshold for noise reduction
    progress=True           # Monitor long-running analysis
)
```

### For Real-Time Analysis:
```python
# Optimize for speed over accuracy
cliff_map = DynamicMap(
    batch_size=50,
    max_iterations=50,       # Fewer iterations
    parallel=True,
    progress=False,         # Disable progress for speed
    verbose=False
)
```

### For High Accuracy:
```python
# Optimize for accuracy over speed
cliff_map = DynamicMap(
    batch_size=30,          # Smaller batches for precision
    max_iterations=300,     # More iterations
    convergence_threshold=1e-6,  # Stricter convergence
    min_samples=5          # Lower threshold for detail
)
```

## Troubleshooting

### Common Issues:

1. **Import Error**: Ensure the package is properly installed or add to PYTHONPATH
2. **Missing Data**: Check that data files exist in the expected locations  
3. **Memory Issues**: Reduce batch_size or enable parallel processing
4. **Slow Performance**: Enable parallel processing and increase n_jobs
5. **Visualization Errors**: Ensure matplotlib backend is properly configured

### Debug Mode:
```python
# Enable verbose output for debugging
cliff_map = DynamicMap(verbose=True, progress=True)
```

### Performance Profiling:
```python
# Use the built-in benchmarking
from examples.complete_demo import performance_benchmark
results = performance_benchmark()
```

## Next Steps

After running the examples:

1. **Customize Parameters**: Adjust CLiFF-map parameters for your specific data
2. **Integrate Checkpointing**: Use checkpointing for long-running analyses
3. **Scale Up**: Apply parallel processing for larger datasets  
4. **Visualize Results**: Create custom visualizations for your use case
5. **Export Results**: Integrate with your analysis pipeline using CSV/XML export

## Support

For issues or questions:
- Check the main package documentation in `../README.md`
- Review test cases in `../tests/test_cliffmap.py`
- Examine the source code in `../cliffmap/`

Enjoy exploring flow patterns with CLiFF-map! 🌊