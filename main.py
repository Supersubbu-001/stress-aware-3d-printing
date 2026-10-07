#!/usr/bin/env python3
"""Main entry point for stress-aware 3D printing pipeline."""

from pathlib import Path
import sys

from src.input_processor import InputProcessor
from src.stress_analyzer import StressAnalyzer
from src.physics_aware_ml import PhysicsAwareMLModel
from src.infill_generator import InfillGenerator
from src.gcode_generator import GCodeGenerator
from src.utils import load_config, ensure_directory, get_logger

logger = get_logger(__name__)


def get_file_path(prompt, file_type="file"):
    """Get file path from user with validation."""
    while True:
        file_path = input(f"\n{prompt}: ").strip()
        if not file_path:
            print("❌ Path cannot be empty. Please try again.")
            continue

        path = Path(file_path)
        if not path.exists():
            print(f"❌ {file_type.capitalize()} not found: {file_path}")
            continue

        if file_type == "file" and not path.is_file():
            print(f"❌ Path is not a file: {file_path}")
            continue

        if file_type == "directory" and not path.is_dir():
            print(f"❌ Path is not a directory: {file_path}")
            continue

        return str(path.resolve())


def get_output_path(prompt):
    """Get output file path from user."""
    while True:
        value = input(f"\n{prompt}: ").strip()
        if not value:
            print("❌ Output path cannot be empty.")
            continue

        path = Path(value)
        ensure_directory(path.parent)

        if path.exists():
            overwrite = input(f"⚠️ File already exists: {path}. Overwrite? (yes/no): ").strip().lower()
            if overwrite not in ("y", "yes"):
                print("Please choose a different path.")
                continue

        return str(path.resolve())


def get_training_choice():
    choice = input("\nDo you want to use a separate CSV for ML training? (yes/no): ").strip().lower()
    if choice in ("y", "yes"):
        return get_file_path("Enter path to training CSV file")
    return None


def build_pipeline(stress_csv: str, stl_path: str, output_path: str):
    config = load_config('config.yaml')

    processor = InputProcessor()
    processor.load_stress_csv(stress_csv)
    processor.load_geometry_stl(stl_path)
    alignment = processor.align_stress_to_mesh()

    stress_values = alignment['stress_per_vertex']
    stress_map = StressAnalyzer(config)
    direct_density = stress_map.compute_density_map(stress_values)

    ml_model = PhysicsAwareMLModel(config)
    train_csv = get_training_choice()
    if train_csv:
        train_processor = InputProcessor()
        train_processor.load_stress_csv(train_csv)
        train_stress = train_processor.stress_data['stress']
        train_density = stress_map.compute_density_map(train_stress)
    else:
        train_stress = stress_values
        train_density = direct_density

    ml_model.train(train_stress, train_density)
    geometry_features = [[1.0, 0.5, 0.25, 0.1] for _ in range(len(stress_values))]
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


def run_interactive():
    print("\n" + "=" * 70)
    print("STRESS-AWARE 3D PRINTING PIPELINE")
    print("=" * 70)

    stress_csv = get_file_path("Enter path to stress CSV file")
    stl_path = get_file_path("Enter path to STL file")
    output_path = get_output_path("Enter path to save the final G-code file")

    return build_pipeline(stress_csv, stl_path, output_path)


if __name__ == '__main__':
    try:
        output_path = run_interactive()
        print(f"\nCreated G-code at: {output_path}")
        print("Pipeline complete.")
    except Exception as exc:
        logger.exception("Pipeline failed")
        print(f"\n❌ Pipeline failed: {exc}")
        sys.exit(1)
