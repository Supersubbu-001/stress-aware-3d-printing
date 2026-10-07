"""Adaptive infill generation for variable density patterns."""

import numpy as np
from .utils import get_logger

logger = get_logger(__name__)


class InfillGenerator:
    """Generate adaptive infill pattern based on density field."""

    def __init__(self, config):
        self.config = config
        self.layer_height = config['print']['layer_height']
        self.nozzle_diameter = config['printer']['nozzle_diameter']

    def build_adaptive_infill(self, mesh_bounds, density_map, layer_count=10):
        """Create a simplified adaptive infill field over the bounding box."""
        min_x, min_y, min_z = mesh_bounds[0]
        max_x, max_y, max_z = mesh_bounds[1]

        density_map = np.asarray(density_map, dtype=float)
        if density_map.size == 0:
            density_map = np.array([50.0])

        mean_density = float(np.mean(density_map))
        if layer_count <= 0:
            layer_count = max(1, int(np.ceil(max_z / self.layer_height)))

        infill_layers = []
        layer_zs = np.linspace(min_z, max_z, num=layer_count)

        for idx, z in enumerate(layer_zs):
            x_coords = np.linspace(min_x, max_x, num=25)
            y_coords = np.linspace(min_y, max_y, num=25)

            density_factor = mean_density / 100.0
            spacing = max(1.5, 12.0 - density_factor * 8.0)

            layer_lines = []
            for y in y_coords:
                line = []
                for x in x_coords:
                    line.append((float(x), float(y), float(z)))
                layer_lines.append(line)

            if idx % 2 == 1:
                layer_lines = [list(reversed(line)) for line in layer_lines]

            infill_layers.append({
                'z': float(z),
                'lines': layer_lines,
                'spacing': spacing,
                'density': mean_density,
            })

        logger.info(f"Generated adaptive infill for {layer_count} layers")
        return infill_layers

    def generate_layer_script(self, layer_data, start_x=0.0, start_y=0.0):
        """Return a list of G1 commands for one layer."""
        commands = []
        for line in layer_data['lines']:
            if len(line) == 0:
                continue
            start = line[0]
            end = line[-1]
            commands.append(f"G1 X{start[0]:.3f} Y{start[1]:.3f} Z{start[2]:.3f} F1200")
            commands.append(f"G1 X{end[0]:.3f} Y{end[1]:.3f} Z{start[2]:.3f} E0.5 F900")
        return commands
