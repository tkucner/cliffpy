"""
Checkpointing and state management utilities for CLiFF-map.

This module provides functionality to save and restore training state,
enabling interrupted training to be resumed efficiently.
"""

import pickle
import os
import json
import datetime
from typing import Dict, Any, Optional
import numpy as np


class CheckpointManager:
    """Manages checkpointing for CLiFF-map training sessions."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.
        
        Parameters:
        -----------
        checkpoint_dir : str, default "checkpoints"
            Directory to store checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        self.ensure_checkpoint_dir()
    
    def ensure_checkpoint_dir(self):
        """Create checkpoint directory if it doesn't exist."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, dynamic_map, checkpoint_name: str = None, 
                       metadata: Dict[str, Any] = None) -> str:
        """
        Save complete training state to checkpoint file.
        
        Parameters:
        -----------
        dynamic_map : DynamicMap
            DynamicMap instance to save
        checkpoint_name : str, optional
            Custom checkpoint name. If None, uses timestamp
        metadata : dict, optional
            Additional metadata to store with checkpoint
        
        Returns:
        --------
        str : Path to saved checkpoint file
        """
        if checkpoint_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"cliff_map_{timestamp}"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pkl")
        
        # Prepare checkpoint data
        checkpoint_data = {
            'dynamic_map_state': self._extract_state(dynamic_map),
            'timestamp': datetime.datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Save checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save human-readable summary
        summary_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_summary.json")
        self._save_summary(checkpoint_data, summary_path)
        
        print(f"Checkpoint saved to: {checkpoint_path}")
        print(f"Summary saved to: {summary_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, dynamic_map=None):
        """
        Load training state from checkpoint file.
        
        Parameters:
        -----------
        checkpoint_path : str
            Path to checkpoint file
        dynamic_map : DynamicMap, optional
            DynamicMap instance to restore state to. If None, creates new instance
        
        Returns:
        --------
        DynamicMap : Restored DynamicMap instance
        dict : Checkpoint metadata
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        if dynamic_map is None:
            from .dynamic_map import DynamicMap
            dynamic_map = DynamicMap()
        
        # Restore state
        self._restore_state(dynamic_map, checkpoint_data['dynamic_map_state'])
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Original timestamp: {checkpoint_data['timestamp']}")
        
        return dynamic_map, checkpoint_data['metadata']
    
    def list_checkpoints(self) -> list:
        """
        List available checkpoint files.
        
        Returns:
        --------
        list : List of checkpoint file paths
        """
        checkpoint_files = []
        
        if os.path.exists(self.checkpoint_dir):
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith('.pkl'):
                    checkpoint_files.append(os.path.join(self.checkpoint_dir, filename))
        
        return sorted(checkpoint_files)
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get information about a checkpoint without fully loading it.
        
        Parameters:
        -----------
        checkpoint_path : str
            Path to checkpoint file
        
        Returns:
        --------
        dict : Checkpoint information
        """
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            state = checkpoint_data['dynamic_map_state']
            
            info = {
                'timestamp': checkpoint_data['timestamp'],
                'metadata': checkpoint_data['metadata'],
                'n_components': len(state.get('components', [])),
                'has_data': state.get('data') is not None,
                'data_shape': state.get('data').shape if state.get('data') is not None else None,
                'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
                'has_history': bool(state.get('history'))
            }
            
            return info
        
        except Exception as e:
            return {'error': str(e)}
    
    def cleanup_old_checkpoints(self, keep_latest: int = 5):
        """
        Remove old checkpoint files, keeping only the most recent ones.
        
        Parameters:
        -----------
        keep_latest : int, default 5
            Number of most recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_latest:
            print(f"Found {len(checkpoints)} checkpoints, keeping all")
            return
        
        # Sort by modification time and remove oldest
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        to_remove = checkpoints[keep_latest:]
        
        for checkpoint_path in to_remove:
            os.remove(checkpoint_path)
            # Also remove summary file if it exists
            summary_path = checkpoint_path.replace('.pkl', '_summary.json')
            if os.path.exists(summary_path):
                os.remove(summary_path)
        
        print(f"Cleaned up {len(to_remove)} old checkpoints, kept {keep_latest} most recent")
    
    def _extract_state(self, dynamic_map) -> Dict[str, Any]:
        """Extract serializable state from DynamicMap instance."""
        state = {}
        
        # Core attributes
        attrs_to_save = [
            'components', 'data', 'batch_size', 'max_iterations',
            'convergence_threshold', 'bandwidth', 'min_samples',
            'parallel', 'n_jobs', 'verbose', 'progress', 'history'
        ]
        
        for attr in attrs_to_save:
            if hasattr(dynamic_map, attr):
                value = getattr(dynamic_map, attr)
                # Convert numpy arrays to ensure they're serializable
                if isinstance(value, np.ndarray):
                    state[attr] = value.copy()
                else:
                    state[attr] = value
        
        return state
    
    def _restore_state(self, dynamic_map, state: Dict[str, Any]):
        """Restore DynamicMap state from saved data."""
        for attr, value in state.items():
            setattr(dynamic_map, attr, value)
    
    def _save_summary(self, checkpoint_data: Dict[str, Any], summary_path: str):
        """Save human-readable checkpoint summary."""
        state = checkpoint_data['dynamic_map_state']
        
        summary = {
            'timestamp': checkpoint_data['timestamp'],
            'metadata': checkpoint_data['metadata'],
            'n_components': len(state.get('components', [])),
            'data_points': state.get('data').shape[0] if state.get('data') is not None else 0,
            'data_dimensions': state.get('data').shape[1] if state.get('data') is not None else 0,
            'settings': {
                'batch_size': state.get('batch_size'),
                'max_iterations': state.get('max_iterations'),
                'convergence_threshold': state.get('convergence_threshold'),
                'bandwidth': state.get('bandwidth'),
                'min_samples': state.get('min_samples'),
                'parallel': state.get('parallel'),
                'n_jobs': state.get('n_jobs')
            }
        }
        
        # Add component details
        if 'components' in state and state['components']:
            summary['components'] = []
            for i, comp in enumerate(state['components']):
                comp_summary = {
                    'id': i,
                    'position': comp.get('position', []).tolist() if isinstance(comp.get('position'), np.ndarray) else comp.get('position'),
                    'direction_rad': float(comp.get('direction', 0)),
                    'direction_deg': float(np.degrees(comp.get('direction', 0))),
                    'weight': float(comp.get('weight', 0)),
                    'uncertainty': float(comp.get('uncertainty', 0))
                }
                summary['components'].append(comp_summary)
        
        # Add training history summary
        if 'history' in state and state['history']:
            history = state['history']
            summary['training_summary'] = {
                'total_iterations': len(history.get('likelihood', [])),
                'final_likelihood': float(history['likelihood'][-1]) if 'likelihood' in history and history['likelihood'] else None,
                'converged': bool(history.get('converged', False)),
                'total_training_time': sum(history.get('processing_time', []))
            }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)


def create_training_session(session_name: str, checkpoint_dir: str = "checkpoints") -> CheckpointManager:
    """
    Create a new training session with checkpointing.
    
    Parameters:
    -----------
    session_name : str
        Name for the training session
    checkpoint_dir : str, default "checkpoints"
        Directory to store checkpoints
    
    Returns:
    --------
    CheckpointManager : Configured checkpoint manager
    """
    session_dir = os.path.join(checkpoint_dir, session_name)
    checkpoint_manager = CheckpointManager(session_dir)
    
    print(f"Training session '{session_name}' created")
    print(f"Checkpoints will be saved to: {session_dir}")
    
    return checkpoint_manager


def auto_checkpoint(dynamic_map, checkpoint_manager: CheckpointManager, 
                   iteration: int, save_every: int = 10) -> bool:
    """
    Automatically save checkpoints at regular intervals during training.
    
    Parameters:
    -----------
    dynamic_map : DynamicMap
        DynamicMap instance being trained
    checkpoint_manager : CheckpointManager
        Checkpoint manager instance
    iteration : int
        Current training iteration
    save_every : int, default 10
        Save checkpoint every N iterations
    
    Returns:
    --------
    bool : True if checkpoint was saved, False otherwise
    """
    if iteration % save_every == 0:
        metadata = {
            'iteration': iteration,
            'auto_checkpoint': True,
            'save_interval': save_every
        }
        checkpoint_manager.save_checkpoint(
            dynamic_map, 
            checkpoint_name=f"auto_iter_{iteration:04d}",
            metadata=metadata
        )
        return True
    
    return False


def resume_training(checkpoint_path: str, dynamic_map=None):
    """
    Resume training from a checkpoint file.
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to checkpoint file
    dynamic_map : DynamicMap, optional
        DynamicMap instance to restore. If None, creates new instance
    
    Returns:
    --------
    DynamicMap : Restored DynamicMap instance
    dict : Training metadata from checkpoint
    """
    checkpoint_manager = CheckpointManager()
    dynamic_map, metadata = checkpoint_manager.load_checkpoint(checkpoint_path, dynamic_map)
    
    print("Training resumed from checkpoint")
    if 'iteration' in metadata:
        print(f"Last iteration: {metadata['iteration']}")
    
    return dynamic_map, metadata