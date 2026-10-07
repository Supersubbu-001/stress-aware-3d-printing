#!/usr/bin/env python3
"""
Stress-aware 3D printing pipeline with interactive user input.
"""

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
            print("   Please provide the full or relative path to the file.")
            continue

        if file_type == "file" and not path.is_file():
            print(f"❌ Path is not a file: {file_path}")
            continue

        if file_type == "directory" and not path.is_dir():
            print(f"❌ Path is not a directory: {file_path}")
            continue

        print(f"✅ {file_type.capitalize()} loaded: {path.absolute()}")
        return str(path.absolute())


def get_output_path(prompt):
    """Get output file path from user."""
    while True:
        output_path = input(f"\n{prompt}: ").strip()

        if not output_path:
            print("❌ Path cannot be empty. Please try again.")
            continue

        path = Path(output_path)
        ensure_directory(path.parent)

        if path.exists():
            overwrite = input(f"⚠️  File already exists: {path}. Overwrite? (yes/no): ").strip().lower()
            if overwrite not in ['yes', 'y']:
                print("Cancelled. Please choose a different path.")
                continue

        print(f"✅ Output will be saved to: {path.absolute()}")
        return str(path.absolute())


def get_training_data():
    """Get training data configuration from user."""
    print("\n" + "=" * 60)
    print("TRAINING DATA CONFIGURATION")
    print("=" * 60)

    use_training = input("\nDo you want to provide separate training data for the ML model? (yes/no, default: no): ").strip().lower()

    if use_training in ['yes', 'y']:
        print("\n📝 Provide training stress data (for model training):")
        training_csv = get_file_path("Enter path to training stress CSV file", "file")
        return training_csv

    print("\n✅ Will use current stress data for training the ML model.")
    return None


def display_welcome():
    """Display welcome message."""
    print("\n" + "=" * 60)
    print("🖨️  STRESS-AWARE 3D PRINTING PIPELINE")
    print("=" * 60)
    print("\nThis pipeline converts ANSYS stress analysis data into adaptive")
    print("infill patterns optimized for your FlashForge Creator 3 Pro printer.")
    print("\nFeatures:")
    print("  • Maps stress data to 3D geometry")
    print("  • Trains ML model on the spot")
    print("  • Generates adaptive infill patterns")
    print("  • Creates printer-ready G-code")
    print("\n" + "=" * 60)


def build_pipeline():
    """Main pipeline with interactive user input."""
    display_welcome()

    try:
        print("\n" + "=" * 60)
        print("INPUT FILES")
        print("=" * 60)

        print("\n📊 Provide stress analysis data:")
        stress_csv = get_file_path("Enter path to stress CSV file (from ANSYS)", "file")

        print("\n📐 Provide 3D geometry:")
        stl_path = get_file_path("Enter path to STL geometry file", "file")

        training_csv = get_training_data()

        print("\n" + "=" * 60)
        print("OUTPUT CONFIGURATION")
        print("=" * 60)
        output_path = get_output_path("Enter path to save G-code file (e.g., output/my_part.gcode)")

        print("\n" + "=" * 60)
        print("LOADING AND PROCESSING DATA")
        print("=" * 60)

        config = load_config("config.yaml")
        print(f"\n✅ Configuration loaded from config.yaml")
        print(f"   Printer: {config['printer']['name']}")
        print(f"   Nozzle temp: {config['print']['nozzle_temp']}°C")
        print(f"   Layer height: {config['print']['layer_height']} mm")

        print("\n📥 Loading input files...")
        processor = InputProcessor()
        processor.load_stress_csv(stress_csv)
        processor.load_geometry_stl(stl_path)
        alignment = processor.align_stress_to_mesh()

        stress_values = alignment["stress_per_vertex"]
        print(f"✅ Aligned {len(stress_values)} stress data points to mesh vertices")

        print("\n🔍 Analyzing stress distribution...")
        stress_map = StressAnalyzer(config)
        direct_density = stress_map.compute_density_map(stress_values)
        print(f"✅ Computed stress-based density map")
        print(f"   Density range: {direct_density.min():.1f}% - {direct_density.max():.1f}%")

        print("\n🤖 Training ML model on the spot...")
        ml_model = PhysicsAwareMLModel(config)

        if training_csv:
            print(f"   Using separate training data from: {training_csv}")
            train_processor = InputProcessor()
            train_processor.load_stress_csv(training_csv)
            train_stress = train_processor.stress_data['stress']
            train_density = stress_map.compute_density_map(train_stress)
        else:
            print("   Using current stress data for training")
            train_stress = stress_values
            train_density = direct_density

        geometry_features = [[1.0, 0.5, 0.25, 0.1] for _ in range(len(train_stress))]
        ml_model.train(train_stress, train_density)
        print("✅ ML model trained successfully")

        print("\n📈 Predicting adaptive infill density...")
        geometry_features = [[1.0, 0.5, 0.25, 0.1] for _ in range(len(stress_values))]
        adaptive_density = ml_model.predict(stress_values, geometry_features)
        print(f"✅ Predicted adaptive density")
        print(f"   Density range: {adaptive_density.min():.1f}% - {adaptive_density.max():.1f}%")

        blend_density = 0.6 * direct_density + 0.4 * adaptive_density
        blend_density = blend_density.astype(float)
        print(f"✅ Blended densities (60% stress-based + 40% ML-predicted)")
        print(f"   Final density range: {blend_density.min():.1f}% - {blend_density.max():.1f}%")

        print("\n🧵 Generating adaptive infill layers...")
        bounds = processor.geometry["bounds"]
        generator = InfillGenerator(config)
        infill_layers = generator.build_adaptive_infill(bounds, blend_density, layer_count=12)
        print("✅ Generated 12 adaptive infill layers")

        print("\n🔧 Generating G-code for FlashForge Creator 3 Pro...")
        gcode_gen = GCodeGenerator(config)
        infill_layers = gcode_gen.build_layer_commands(infill_layers)
        output_file = gcode_gen.generate(infill_layers, output_path)

        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 60)
        print("\n✨ G-code file created successfully!")
        print(f"   Location: {output_file}")
        print(f"   File size: {output_file.stat().st_size} bytes")
        print("\n📋 Next steps:")
        print("   1. Copy the G-code file to a USB drive")
        print("   2. Insert USB into FlashForge Creator 3 Pro")
        print("   3. Select file and print from printer touchscreen")
        print("\n" + "=" * 60 + "\n")

        return output_file

    except FileNotFoundError as e:
        print(f"\n❌ File error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Data error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    build_pipeline()
