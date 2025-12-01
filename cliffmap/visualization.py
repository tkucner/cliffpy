"""
Visualization utilities for CLiFF-map flow field analysis.

This module provides comprehensive visualization tools for dynamic flow field
maps, including component visualization, flow direction plots, and data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
import seaborn as sns


class FlowFieldVisualizer:
    """Visualization tools for flow field analysis."""
    
    def __init__(self, figsize=(12, 8), dpi=100):
        """Initialize visualizer with default figure settings."""
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use('seaborn-v0_8')
    
    def plot_components(self, dynamic_map, save_path=None, show_ellipses=True):
        """
        Plot learned components with flow directions and uncertainty ellipses.
        
        Parameters:
        -----------
        dynamic_map : DynamicMap
            Fitted DynamicMap instance
        save_path : str, optional
            Path to save the figure
        show_ellipses : bool, default True
            Whether to show uncertainty ellipses
        
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot original data points
        if hasattr(dynamic_map, 'data') and dynamic_map.data is not None:
            ax.scatter(dynamic_map.data[:, 0], dynamic_map.data[:, 1], 
                      alpha=0.3, s=20, c='lightgray', label='Data points')
        
        # Plot components
        colors = plt.cm.Set3(np.linspace(0, 1, len(dynamic_map.components)))
        
        for i, component in enumerate(dynamic_map.components):
            x, y = component['position']
            direction = component['direction']
            weight = component['weight']
            uncertainty = component.get('uncertainty', 0.1)
            
            # Plot component center
            ax.scatter(x, y, s=weight*500, c=[colors[i]], 
                      alpha=0.8, edgecolors='black', linewidth=1,
                      label=f'Component {i+1} (w={weight:.3f})')
            
            # Plot flow direction arrow
            dx = 0.5 * np.cos(direction)
            dy = 0.5 * np.sin(direction)
            ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1,
                    fc=colors[i], ec='black', alpha=0.8, linewidth=2)
            
            # Plot uncertainty ellipse
            if show_ellipses and uncertainty > 0:
                circle = Circle((x, y), uncertainty, 
                              color=colors[i], alpha=0.2, linewidth=1)
                ax.add_patch(circle)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('CLiFF-map Flow Field Components')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig, ax
    
    def plot_flow_field(self, dynamic_map, resolution=50, save_path=None):
        """
        Plot interpolated flow field over the entire domain.
        
        Parameters:
        -----------
        dynamic_map : DynamicMap
            Fitted DynamicMap instance
        resolution : int, default 50
            Grid resolution for interpolation
        save_path : str, optional
            Path to save the figure
        
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        if not hasattr(dynamic_map, 'components') or not dynamic_map.components:
            raise ValueError("DynamicMap must be fitted before plotting flow field")
        
        # Create interpolation grid
        if hasattr(dynamic_map, 'data') and dynamic_map.data is not None:
            x_min, x_max = dynamic_map.data[:, 0].min() - 1, dynamic_map.data[:, 0].max() + 1
            y_min, y_max = dynamic_map.data[:, 1].min() - 1, dynamic_map.data[:, 1].max() + 1
        else:
            # Use component positions to determine bounds
            positions = np.array([c['position'] for c in dynamic_map.components])
            x_min, x_max = positions[:, 0].min() - 2, positions[:, 0].max() + 2
            y_min, y_max = positions[:, 1].min() - 2, positions[:, 1].max() + 2
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Interpolate flow directions
        directions = self._interpolate_flow(dynamic_map, grid_points)
        u = np.cos(directions).reshape(xx.shape)
        v = np.sin(directions).reshape(xx.shape)
        
        # Calculate flow magnitude (based on component weights)
        magnitudes = self._interpolate_magnitude(dynamic_map, grid_points)
        magnitude_grid = magnitudes.reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot flow field
        quiver = ax.quiver(xx, yy, u, v, magnitude_grid, 
                          cmap='viridis', alpha=0.7, scale=20)
        plt.colorbar(quiver, ax=ax, label='Flow Magnitude')
        
        # Overlay components
        for i, component in enumerate(dynamic_map.components):
            x, y = component['position']
            ax.scatter(x, y, s=200, c='red', marker='x', linewidth=3,
                      label='Components' if i == 0 else "")
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('CLiFF-map Interpolated Flow Field')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Flow field saved to: {save_path}")
        
        return fig, ax
    
    def plot_training_history(self, dynamic_map, save_path=None):
        """
        Plot training convergence history if available.
        
        Parameters:
        -----------
        dynamic_map : DynamicMap
            Fitted DynamicMap instance
        save_path : str, optional
            Path to save the figure
        
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        if not hasattr(dynamic_map, 'history') or not dynamic_map.history:
            print("No training history available")
            return None, None
        
        history = dynamic_map.history
        
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0], self.figsize[1]*1.2), dpi=self.dpi)
        axes = axes.flatten()
        
        # Plot likelihood evolution
        if 'likelihood' in history:
            axes[0].plot(history['likelihood'])
            axes[0].set_title('Log-Likelihood Evolution')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Log-Likelihood')
            axes[0].grid(True, alpha=0.3)
        
        # Plot number of components
        if 'n_components' in history:
            axes[1].plot(history['n_components'])
            axes[1].set_title('Number of Components')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Components Count')
            axes[1].grid(True, alpha=0.3)
        
        # Plot convergence metrics
        if 'convergence' in history:
            axes[2].semilogy(history['convergence'])
            axes[2].set_title('Convergence Metric')
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('Change (log scale)')
            axes[2].grid(True, alpha=0.3)
        
        # Plot processing times
        if 'processing_time' in history:
            axes[3].plot(np.cumsum(history['processing_time']))
            axes[3].set_title('Cumulative Processing Time')
            axes[3].set_xlabel('Iteration')
            axes[3].set_ylabel('Time (seconds)')
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Training history saved to: {save_path}")
        
        return fig, axes
    
    def compare_results(self, results_dict, save_path=None):
        """
        Compare results from different methods or parameters.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary with method names as keys and DynamicMap instances as values
        save_path : str, optional
            Path to save the figure
        
        Returns:
        --------
        fig, axes : matplotlib figure and axis objects
        """
        n_methods = len(results_dict)
        fig, axes = plt.subplots(1, n_methods, figsize=(self.figsize[0]*n_methods/2, self.figsize[1]), dpi=self.dpi)
        
        if n_methods == 1:
            axes = [axes]
        
        for i, (method_name, dynamic_map) in enumerate(results_dict.items()):
            ax = axes[i]
            
            # Plot components
            colors = plt.cm.Set3(np.linspace(0, 1, len(dynamic_map.components)))
            
            for j, component in enumerate(dynamic_map.components):
                x, y = component['position']
                direction = component['direction']
                weight = component['weight']
                
                # Plot component center
                ax.scatter(x, y, s=weight*500, c=[colors[j]], 
                          alpha=0.8, edgecolors='black', linewidth=1)
                
                # Plot flow direction arrow
                dx = 0.5 * np.cos(direction)
                dy = 0.5 * np.sin(direction)
                ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1,
                        fc=colors[j], ec='black', alpha=0.8, linewidth=2)
            
            ax.set_title(f'{method_name}\n({len(dynamic_map.components)} components)')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        return fig, axes
    
    def _interpolate_flow(self, dynamic_map, points):
        """Interpolate flow directions at given points using component weights."""
        directions = np.zeros(len(points))
        
        for point_idx, point in enumerate(points):
            weighted_directions = []
            total_weight = 0
            
            for component in dynamic_map.components:
                pos = component['position']
                direction = component['direction']
                weight = component['weight']
                
                # Calculate distance-based weight
                dist = np.linalg.norm(point - pos)
                influence = weight * np.exp(-dist**2 / 2)
                
                weighted_directions.append(influence * np.array([np.cos(direction), np.sin(direction)]))
                total_weight += influence
            
            if total_weight > 1e-10:
                # Average weighted directions
                avg_direction = np.sum(weighted_directions, axis=0) / total_weight
                directions[point_idx] = np.arctan2(avg_direction[1], avg_direction[0])
            else:
                directions[point_idx] = 0
        
        return directions
    
    def _interpolate_magnitude(self, dynamic_map, points):
        """Interpolate flow magnitudes at given points."""
        magnitudes = np.zeros(len(points))
        
        for point_idx, point in enumerate(points):
            total_influence = 0
            
            for component in dynamic_map.components:
                pos = component['position']
                weight = component['weight']
                
                # Calculate distance-based influence
                dist = np.linalg.norm(point - pos)
                influence = weight * np.exp(-dist**2 / 2)
                total_influence += influence
            
            magnitudes[point_idx] = total_influence
        
        return magnitudes


def plot_data_distribution(data, save_path=None, bins=50):
    """
    Plot data distribution with marginal histograms.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data (N, 2) for position or (N, 4) for position+direction
    save_path : str, optional
        Path to save the figure
    bins : int, default 50
        Number of histogram bins
    
    Returns:
    --------
    fig : matplotlib figure object
    """
    if data.shape[1] >= 4:
        # Position and direction data
        fig = plt.figure(figsize=(15, 5))
        
        # Position distribution
        ax1 = plt.subplot(131)
        plt.scatter(data[:, 0], data[:, 1], alpha=0.6, s=10)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Spatial Distribution')
        plt.grid(True, alpha=0.3)
        
        # Direction histogram
        ax2 = plt.subplot(132)
        angles_deg = np.degrees(data[:, 2]) % 360
        plt.hist(angles_deg, bins=bins, alpha=0.7, edgecolor='black')
        plt.xlabel('Direction (degrees)')
        plt.ylabel('Frequency')
        plt.title('Direction Distribution')
        plt.grid(True, alpha=0.3)
        
        # Speed distribution (if available)
        if data.shape[1] > 3:
            ax3 = plt.subplot(133)
            plt.hist(data[:, 3], bins=bins, alpha=0.7, edgecolor='black')
            plt.xlabel('Speed')
            plt.ylabel('Frequency')
            plt.title('Speed Distribution')
            plt.grid(True, alpha=0.3)
        
    else:
        # Position only
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], alpha=0.6, s=10)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Data Distribution')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Data distribution plot saved to: {save_path}")
    
    return fig