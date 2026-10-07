#!/usr/bin/env python3
"""Main entry point for stress-aware 3D printing pipeline."""

import argparse
from pathlib import Path

from src.input_processor import InputProcessor
from src.stress_analyzer import StressAnalyzer
from src.physics_aware_ml import PhysicsAwareMLModel
from src.infill_generator import InfillGenerator
from src.gcode_generator import GCodeGenerator
from src.utils import load_config, ensure_directory, get_logger

logger = get_logger(__name__)


def build_pipeline(stress_csv: str, stl_path: str, output_path: str):
    config = load_config('config.yaml')

    processor = InputProcessor()
    processor.load_stress_csv(stress_csv)
    processor.load_geometry_stl(stl_path)
    alignment = processor.align_stress_to_mesh()

    stress_values = alignment['stress_per_vertex']
    stress_map = StressAnalyzer(config)
    direct_density = stress_map.compute_density_map(stress_values)

    geometry_features = [[1.0, 0.5, 0.25, 0.1] for _ in range(len(stress_values))]
    ml_model = PhysicsAwareMLModel(config)
    adaptive_density = ml_model.predict(stress_values, geometry_features)

    blend_density = 0.6 * direct_density + 0.4 * adaptive_density
    blend_density = blend_density.astype(float)

    bounds = processor.geometry['bounds']
    generator = InfillGenerator(config)
    infill_layers = generator.build_adaptive_infill(bounds, blend_density, layer_count=12)
    infill_layers = generator.build_layer_script(infill_layers)

    gcode_gen = GCodeGenerator(config)
    infill_layers = gcode_gen.build_layer_commands(infill_layers)
    output_file = output_path or 'data/outputs/stress_aware_part.gcode'
    ensure_directory(Path(output_file).parent)
    gcode_gen.generate(infill_layers, output_file)

    logger.info('Pipeline complete. Generated adaptive G-code: %s', output_file)
    return output_file


def parse_args():
    parser = argparse.ArgumentParser(description='Stress-aware 3D printing adaptive infill generator')
    parser.add_argument('--stress-data', required=True, help='ANSYS stress CSV file')
    parser.add_argument('--geometry', required=True, help='STL geometry file')
    parser.add_argument('--output', default='data/outputs/stress_aware_part.gcode', help='Output G-code path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build_pipeline(args.stress_data, args.geometry, args.output)
    print(f'Created G-code at: {args.output}')
    print('Pipeline complete.')
