"""
Utility functions for stress-aware 3D printing
"""

import numpy as np
import logging
from pathlib import Path
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger(name):
    """Get a configured logger instance"""
    return logging.getLogger(name)

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, config_path="config.yaml"):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def normalize_stress(stress_values, method="minmax"):
    """
    Normalize stress values to [0, 1] range
    
    Args:
        stress_values: numpy array of stress values
        method: 'minmax' or 'zscore'
    
    Returns:
        normalized stress values
    """
    if method == "minmax":
        stress_min = np.min(stress_values)
        stress_max = np.max(stress_values)
        if stress_max == stress_min:
            return np.ones_like(stress_values) * 0.5
        return (stress_values - stress_min) / (stress_max - stress_min)
    
    elif method == "zscore":
        stress_mean = np.mean(stress_values)
        stress_std = np.std(stress_values)
        if stress_std == 0:
            return np.zeros_like(stress_values)
        return (stress_values - stress_mean) / stress_std
    
    return stress_values

def stress_to_density(stress_normalized, min_density=10, max_density=100):
    """
    Convert normalized stress values to infill density percentages
    
    Args:
        stress_normalized: stress values in [0, 1] range
        min_density: minimum infill density %
        max_density: maximum infill density %
    
    Returns:
        infill density percentages
    """
    density = min_density + stress_normalized * (max_density - min_density)
    return np.clip(density, min_density, max_density)

def create_density_zones(density_map, num_zones=5):
    """
    Create discrete density zones from continuous density map
    
    Args:
        density_map: continuous density values [0-100]
        num_zones: number of discrete zones
    
    Returns:
        zone assignments for each point
    """
    zone_edges = np.linspace(0, 100, num_zones + 1)
    zones = np.digitize(density_map, zone_edges) - 1
    zones = np.clip(zones, 0, num_zones - 1)
    return zones

def interpolate_field(points, values, query_points, method="linear"):
    """
    Interpolate scalar field from known points to query points
    
    Args:
        points: (N, 3) array of known point coordinates
        values: (N,) array of values at known points
        query_points: (M, 3) array of query point coordinates
        method: 'linear' (rbf) or 'nearest'
    
    Returns:
        (M,) array of interpolated values
    """
    from scipy.interpolate import Rbf, NearestNDInterpolator
    
    if method == "linear":
        # Radial basis function interpolation
        rbf = Rbf(points[:, 0], points[:, 1], points[:, 2], values, function='thin_plate')
        return rbf(query_points[:, 0], query_points[:, 1], query_points[:, 2])
    
    elif method == "nearest":
        interp = NearestNDInterpolator(points, values)
        return interp(query_points)
    
    return values

def smooth_density_map(density_map, kernel_size=3, iterations=1):
    """
    Smooth density map using gaussian filtering
    
    Args:
        density_map: 3D or flattened density map
        kernel_size: smoothing kernel size
        iterations: number of smoothing iterations
    
    Returns:
        smoothed density map
    """
    from scipy.ndimage import gaussian_filter
    
    smoothed = density_map.copy()
    for _ in range(iterations):
        smoothed = gaussian_filter(smoothed, sigma=kernel_size/3)
    
    return smoothed

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return Path(directory)

def calculate_print_time(gcode_lines, print_speed=60):
    """
    Estimate print time from G-code
    
    Args:
        gcode_lines: list of G-code commands
        print_speed: default print speed in mm/s
    
    Returns:
        estimated print time in seconds
    """
    total_distance = 0
    current_speed = print_speed
    
    for line in gcode_lines:
        line = line.strip()
        if line.startswith('G1') or line.startswith('G0'):
            # Parse X, Y, Z, F from line
            params = {}
            parts = line.split()
            for part in parts:
                if part[0] in 'XYZF':
                    try:
                        params[part[0]] = float(part[1:])
                    except:
                        pass
            
            if 'F' in params:
                current_speed = params['F'] / 60  # Convert mm/min to mm/s
    
    return total_distance / current_speed if current_speed > 0 else 0

def calculate_material_weight(gcode_lines, filament_diameter=1.75, material_density=1.25):
    """
    Estimate material weight from G-code
    
    Args:
        gcode_lines: list of G-code commands
        filament_diameter: diameter in mm
        material_density: density in g/cm³
    
    Returns:
        estimated material weight in grams
    """
    filament_radius = filament_diameter / 2
    filament_area = np.pi * (filament_radius ** 2)  # mm²
    
    total_length = 0
    for line in gcode_lines:
        if line.strip().startswith('G1') and 'E' in line:
            try:
                e_value = float(line.split('E')[1].split()[0])
                total_length += abs(e_value)
            except:
                pass
    
    # Volume in mm³
    volume = total_length * filament_area
    # Weight in grams (convert mm³ to cm³)
    weight = volume * material_density / 1000
    
    return weight

def log_summary(logger, stress_data, density_map, gcode_lines):
    """Log summary statistics"""
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Stress Statistics:")
    logger.info(f"  Min: {np.min(stress_data):.2f} MPa")
    logger.info(f"  Max: {np.max(stress_data):.2f} MPa")
    logger.info(f"  Mean: {np.mean(stress_data):.2f} MPa")
    logger.info(f"  Std Dev: {np.std(stress_data):.2f} MPa")
    
    logger.info(f"Infill Density Statistics:")
    logger.info(f"  Min: {np.min(density_map):.1f}%")
    logger.info(f"  Max: {np.max(density_map):.1f}%")
    logger.info(f"  Mean: {np.mean(density_map):.1f}%")
    
    logger.info(f"G-code Statistics:")
    logger.info(f"  Total Lines: {len(gcode_lines)}")
    logger.info("=" * 60)
