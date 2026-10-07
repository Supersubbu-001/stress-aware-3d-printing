"""
Stress analysis pipeline for ANSYS data to adaptive infill G-code
"""

import numpy as np
from .utils import get_logger

logger = get_logger(__name__)


class StressAnalyzer:
    """Convert stress field into density-aware infill targets."""

    def __init__(self, config):
        self.config = config
        self.min_density = config['infill_optimization']['min_infill_density']
        self.max_density = config['infill_optimization']['max_infill_density']

    def normalize_stress(self, stress_values):
        stress = np.asarray(stress_values, dtype=float)
        if stress.size == 0:
            return stress
        smin = np.min(stress)
        smax = np.max(stress)
        if np.isclose(smax, smin):
            return np.ones_like(stress) * 0.5
        return (stress - smin) / (smax - smin)

    def compute_density_map(self, stress_values):
        normalized = self.normalize_stress(stress_values)
        density = self.min_density + normalized * (self.max_density - self.min_density)
        density = np.clip(density, self.min_density, self.max_density)
        return density

    def summarize(self, stress_values):
        stress = np.asarray(stress_values, dtype=float)
        return {
            'min': float(np.min(stress)),
            'max': float(np.max(stress)),
            'mean': float(np.mean(stress)),
            'std': float(np.std(stress)),
        }
