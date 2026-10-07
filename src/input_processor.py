"""
Input processor for ANSYS stress data (CSV) and STL geometry
"""

import numpy as np
import pandas as pd
import trimesh
from pathlib import Path
from .utils import get_logger

logger = get_logger(__name__)

class InputProcessor:
    """Process ANSYS stress CSV and STL geometry files"""
    
    def __init__(self):
        self.stress_data = None
        self.geometry = None
        self.mesh = None
        
    def load_stress_csv(self, csv_path):
        """
        Load ANSYS stress data from CSV file
        
        Expected columns: node_id, x, y, z, stress_mpa, strain
        
        Args:
            csv_path: path to CSV file
        
        Returns:
            dict with stress data and coordinates
        """
        logger.info(f"Loading stress data from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} stress data points")
            
            # Validate required columns
            required_cols = ['x', 'y', 'z', 'stress_mpa']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            self.stress_data = {
                'coordinates': df[['x', 'y', 'z']].values,
                'stress': df['stress_mpa'].values,
                'node_ids': df.get('node_id', np.arange(len(df))).values,
                'strain': df.get('strain', np.zeros(len(df))).values,
                'raw_df': df
            }
            
            logger.info(f"Stress range: {self.stress_data['stress'].min():.2f} - {self.stress_data['stress'].max():.2f} MPa")
            
            return self.stress_data
        
        except Exception as e:
            logger.error(f"Error loading stress CSV: {e}")
            raise
    
    def load_geometry_stl(self, stl_path):
        """
        Load 3D geometry from STL file
        
        Args:
            stl_path: path to STL file
        
        Returns:
            trimesh object with geometry
        """
        logger.info(f"Loading geometry from {stl_path}")
        
        try:
            mesh = trimesh.load(stl_path)
            logger.info(f"Loaded mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
            # Validate mesh
            if not mesh.is_valid:
                logger.warning("Mesh is not valid, attempting repair...")
                mesh.remove_duplicate_faces()
                mesh.remove_degenerate_faces()
            
            self.mesh = mesh
            self.geometry = {
                'vertices': mesh.vertices,
                'faces': mesh.faces,
                'mesh': mesh,
                'bounds': mesh.bounds,
                'center': mesh.center_mass,
                'volume': mesh.volume
            }
            
            logger.info(f"Geometry bounds: {self.geometry['bounds']}")
            logger.info(f"Part volume: {self.geometry['volume']:.2f} mm³")
            
            return self.geometry
        
        except Exception as e:
            logger.error(f"Error loading STL file: {e}")
            raise
    
    def validate_inputs(self):
        """Validate that both stress data and geometry are loaded"""
        if self.stress_data is None:
            raise ValueError("Stress data not loaded. Call load_stress_csv() first.")
        if self.geometry is None:
            raise ValueError("Geometry not loaded. Call load_geometry_stl() first.")
        
        logger.info("✓ Both stress data and geometry validated successfully")
        return True
    
    def align_stress_to_mesh(self):
        """
        Align stress data to mesh vertices
        
        Creates mapping between stress data points and nearest mesh vertices
        
        Returns:
            dict with alignment information
        """
        if not self.validate_inputs():
            return None
        
        logger.info("Aligning stress data to mesh vertices...")
        
        from scipy.spatial import cKDTree
        
        stress_coords = self.stress_data['coordinates']
        mesh_vertices = self.geometry['vertices']
        
        # Build KD-tree for fast nearest neighbor search
        tree = cKDTree(mesh_vertices)
        distances, indices = tree.query(stress_coords, k=1)
        
        # Map stress values to mesh vertices
        vertex_stress = np.zeros(len(mesh_vertices))
        vertex_count = np.zeros(len(mesh_vertices))
        
        for idx, stress_val in zip(indices, self.stress_data['stress']):
            vertex_stress[idx] += stress_val
            vertex_count[idx] += 1
        
        # Average multiple stress values at same vertex
        valid = vertex_count > 0
        vertex_stress[valid] /= vertex_count[valid]
        vertex_stress[~valid] = np.mean(self.stress_data['stress'])
        
        logger.info(f"Mapped stress to {np.sum(valid)} mesh vertices")
        
        alignment = {
            'stress_per_vertex': vertex_stress,
            'alignment_distances': distances,
            'alignment_indices': indices
        }
        
        return alignment
    
    def get_stress_statistics(self):
        """Get statistics of loaded stress data"""
        if self.stress_data is None:
            raise ValueError("Stress data not loaded")
        
        stress = self.stress_data['stress']
        
        stats = {
            'min': np.min(stress),
            'max': np.max(stress),
            'mean': np.mean(stress),
            'std': np.std(stress),
            'median': np.median(stress),
            'q25': np.percentile(stress, 25),
            'q75': np.percentile(stress, 75),
            'points': len(stress)
        }
        
        return stats
    
    def get_geometry_statistics(self):
        """Get statistics of loaded geometry"""
        if self.geometry is None:
            raise ValueError("Geometry not loaded")
        
        stats = {
            'vertices': len(self.geometry['vertices']),
            'faces': len(self.geometry['faces']),
            'volume': self.geometry['volume'],
            'bounds': self.geometry['bounds'].tolist(),
            'center': self.geometry['center'].tolist(),
            'is_watertight': self.mesh.is_watertight
        }
        
        return stats
    
    def resample_stress_data(self, target_points=1000):
        """
        Resample stress data to a specific number of points
        
        Args:
            target_points: desired number of points
        
        Returns:
            resampled stress data
        """
        if self.stress_data is None:
            raise ValueError("Stress data not loaded")
        
        current_points = len(self.stress_data['stress'])
        
        if current_points == target_points:
            return self.stress_data
        
        logger.info(f"Resampling stress data from {current_points} to {target_points} points...")
        
        from scipy.interpolate import griddata
        
        coords = self.stress_data['coordinates']
        stress = self.stress_data['stress']
        
        # Create uniform grid for resampling
        bounds = np.array([coords.min(axis=0), coords.max(axis=0)])
        
        # Simple random resampling for demonstration
        indices = np.random.choice(len(coords), target_points, replace=len(coords) < target_points)
        
        resampled = {
            'coordinates': coords[indices],
            'stress': stress[indices],
            'node_ids': self.stress_data['node_ids'][indices]
        }
        
        logger.info(f"Resampling complete")
        return resampled
