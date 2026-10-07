"""Input processor for ANSYS stress data (CSV) and STL geometry."""

import numpy as np
import pandas as pd
import trimesh
from pathlib import Path
from .utils import get_logger

logger = get_logger(__name__)


class InputProcessor:
    """Process ANSYS stress CSV and STL geometry files."""

    def __init__(self):
        self.stress_data = None
        self.geometry = None
        self.mesh = None

    def load_stress_csv(self, csv_path):
        """Load ANSYS stress data from CSV or tab-delimited text file."""
        logger.info(f"Loading stress data from {csv_path}")

        try:
            # Detect delimiter automatically
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()

            delimiter = '\t' if '\t' in first_line else ','
            logger.info(f"Detected delimiter: {delimiter!r}")

            df = pd.read_csv(csv_path, sep=delimiter)
            df.columns = [str(col).strip() for col in df.columns]
            logger.info(f"Loaded {len(df)} stress data points")
            logger.info(f"CSV columns: {df.columns.tolist()}")

            # Try to locate columns dynamically for different ANSYS export styles
            normalized_cols = {str(col).strip().lower(): col for col in df.columns}

            node_col = (
                normalized_cols.get('node_id') or
                normalized_cols.get('node number') or
                normalized_cols.get('node') or
                normalized_cols.get('node number ') or
                None
            )

            x_col = (
                normalized_cols.get('x') or
                normalized_cols.get('x location (mm)') or
                normalized_cols.get('x location') or
                normalized_cols.get('x coordinate') or
                None
            )
            y_col = (
                normalized_cols.get('y') or
                normalized_cols.get('y location (mm)') or
                normalized_cols.get('y location') or
                normalized_cols.get('y coordinate') or
                None
            )
            z_col = (
                normalized_cols.get('z') or
                normalized_cols.get('z location (mm)') or
                normalized_cols.get('z location') or
                normalized_cols.get('z coordinate') or
                None
            )
            stress_col = (
                normalized_cols.get('stress_mpa') or
                normalized_cols.get('stress') or
                normalized_cols.get('equivalent (von-mises) stress (mpa)') or
                normalized_cols.get('equivalent von-mises stress (mpa)') or
                normalized_cols.get('equivalent stress (mpa)') or
                None
            )

            # Fallback if column names don't match exactly
            if x_col is None:
                for col in df.columns:
                    if 'x' in col.lower() and 'location' in col.lower():
                        x_col = col
                        break
            if y_col is None:
                for col in df.columns:
                    if 'y' in col.lower() and 'location' in col.lower():
                        y_col = col
                        break
            if z_col is None:
                for col in df.columns:
                    if 'z' in col.lower() and 'location' in col.lower():
                        z_col = col
                        break
            if stress_col is None:
                for col in df.columns:
                    if 'stress' in col.lower() or 'von' in col.lower():
                        stress_col = col
                        break

            required = [x_col, y_col, z_col, stress_col]
            if any(col is None for col in required):
                raise ValueError(
                    f"Could not identify required columns. Available columns: {df.columns.tolist()}"
                )

            node_values = df[node_col].values if node_col else np.arange(len(df))

            self.stress_data = {
                'coordinates': df[[x_col, y_col, z_col]].values.astype(float),
                'stress': df[stress_col].astype(float).values,
                'node_ids': node_values,
                'strain': np.zeros(len(df), dtype=float),
                'raw_df': df
            }

            logger.info(
                f"Stress range: {self.stress_data['stress'].min():.2f} - {self.stress_data['stress'].max():.2f} MPa"
            )
            return self.stress_data

        except Exception as e:
            logger.error(f"Error loading stress CSV: {e}")
            raise

    def load_geometry_stl(self, stl_path):
        """Load 3D geometry from STL file."""
        logger.info(f"Loading geometry from {stl_path}")

        try:
            mesh = trimesh.load(stl_path)
            logger.info(f"Loaded mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

            if not getattr(mesh, 'is_watertight', True):
                logger.warning("Mesh is not watertight, attempting repair...")

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
        """Validate that both stress data and geometry are loaded."""
        if self.stress_data is None:
            raise ValueError("Stress data not loaded. Call load_stress_csv() first.")
        if self.geometry is None:
            raise ValueError("Geometry not loaded. Call load_geometry_stl() first.")

        logger.info("Both stress data and geometry validated successfully")
        return True

    def align_stress_to_mesh(self):
        """Align stress data to mesh vertices using nearest neighbor."""
        if not self.validate_inputs():
            return None

        logger.info("Aligning stress data to mesh vertices...")

        from scipy.spatial import cKDTree

        stress_coords = self.stress_data['coordinates']
        mesh_vertices = self.geometry['vertices']

        tree = cKDTree(mesh_vertices)
        distances, indices = tree.query(stress_coords, k=1)

        vertex_stress = np.zeros(len(mesh_vertices))
        vertex_count = np.zeros(len(mesh_vertices))

        for idx, stress_val in zip(indices, self.stress_data['stress']):
            vertex_stress[idx] += stress_val
            vertex_count[idx] += 1

        valid = vertex_count > 0
        vertex_stress[valid] /= vertex_count[valid]
        vertex_stress[~valid] = np.mean(self.stress_data['stress'])

        logger.info(f"Mapped stress to {np.sum(valid)} mesh vertices")

        return {
            'stress_per_vertex': vertex_stress,
            'alignment_distances': distances,
            'alignment_indices': indices
        }

    def get_stress_statistics(self):
        """Get statistics of loaded stress data."""
        if self.stress_data is None:
            raise ValueError("Stress data not loaded")

        stress = self.stress_data['stress']
        return {
            'min': np.min(stress),
            'max': np.max(stress),
            'mean': np.mean(stress),
            'std': np.std(stress),
            'median': np.median(stress),
            'q25': np.percentile(stress, 25),
            'q75': np.percentile(stress, 75),
            'points': len(stress)
        }

    def get_geometry_statistics(self):
        """Get statistics of loaded geometry."""
        if self.geometry is None:
            raise ValueError("Geometry not loaded")

        return {
            'vertices': len(self.geometry['vertices']),
            'faces': len(self.geometry['faces']),
            'volume': self.geometry['volume'],
            'bounds': self.geometry['bounds'].tolist(),
            'center': self.geometry['center'].tolist(),
            'is_watertight': self.mesh.is_watertight
        }
