#!/usr/bin/env python3
"""
YOLO Dataset Quality Analysis Tool
Main script for analyzing YOLO datasets using FiftyOne

Author: Alexis Béduneau
License: MIT
"""

import os
import argparse
import yaml
from typing import Dict, Any

import fiftyone as fo
import fiftyone.zoo as foz

from src.config import get_box_field_from_task
from src.dataset import prepare_voxel_dataset
from src.enum import DatasetTask
from src.images import generate_thumbnails
from src.voxel51 import compute_visualizations


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze YOLO dataset quality using FiftyOne",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file (optional)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (optional)"
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to YOLO dataset (required)"
    )
    
    parser.add_argument(
        "--dataset-task",
        type=str,
        required=True,
        choices=["classification", "detection", "segmentation", "pose", "obb"],
        help="Dataset task type (required)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name for FiftyOne dataset (default: auto-generated from path)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Force reload dataset even if it exists"
    )
    
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        default=False,
        help="Skip embedding computation"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding computation"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="clip-vit-base32-torch",
        help="CLIP model name for embeddings"
    )
    
    parser.add_argument(
        "--thumbnail-dir",
        type=str,
        default="thumbnails",
        help="Base directory for thumbnails"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=5151,
        help="Port for FiftyOne app"
    )
    
    parser.add_argument(
        "--no-launch",
        action="store_true",
        default=False,
        help="Don't launch FiftyOne app after processing"
    )
    
    args = parser.parse_args()
    
    # Auto-generate dataset name if not provided
    if args.dataset_name is None:
        import pathlib

        dataset_path = pathlib.Path(args.dataset_path)
        args.dataset_name = (
            dataset_path.parent.name if not dataset_path.is_dir() else dataset_path.name
        )

    return args


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build configuration from arguments and optional config file"""
    
    # Start with default structure
    config = {
        'dataset': {
            'path': args.dataset_path,
            'name': args.dataset_name,
            'task': args.dataset_task,
            'reload': args.reload
        },
        'embeddings': {
            'skip': args.skip_embeddings,
            'model': args.model,
            'batch_size': args.batch_size
        },
        'thumbnail_dir': args.thumbnail_dir,
        'port': args.port,
        'no_launch': args.no_launch
    }
    
    # If config file is provided, load it and override with args
    if args.config and os.path.exists(args.config):
        try:
            file_config = load_config(args.config)
            
            # Merge file config with args (args take precedence)
            if 'dataset' in file_config:
                config['dataset']['path'] = args.dataset_path  # Always use args
                config['dataset']['task'] = args.dataset_task  # Always use args
                config['dataset']['name'] = args.dataset_name  # Always use args
                config['dataset']['reload'] = args.reload or file_config['dataset'].get('reload', False)
            
            if 'embeddings' in file_config:
                # Only override if args were explicitly set (not defaults)
                if not args.skip_embeddings:
                    config['embeddings']['skip'] = file_config['embeddings'].get('skip', False)
                if args.batch_size == 16:  # Default value
                    config['embeddings']['batch_size'] = file_config['embeddings'].get('batch_size', 16)
                if args.model == "clip-vit-base32-torch":  # Default value
                    config['embeddings']['model'] = file_config['embeddings'].get('model', "clip-vit-base32-torch")

        except Exception as e:
            print(f"⚠️  Warning: Could not load config file {args.config}: {e}")
            print("Using command-line arguments only")
    
    return config


def main():
    """Main execution function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Build configuration
    config = build_config(args)
    
    # Validate dataset path
    if not os.path.exists(config['dataset']['path']):
        print(f"❌ Dataset path does not exist: {config['dataset']['path']}")
        return
    
    print("\n" + "=" * 60)
    print("FIFTYONE YOLO DATASET ANALYSIS")
    print("=" * 60)
    print(f"📁 Dataset Path: {config['dataset']['path']}")
    print(f"📊 Dataset Name: {config['dataset']['name']}")
    print(f"🎯 Dataset Task: {config['dataset']['task']}")
    print(f"🔄 Force Reload: {config['dataset']['reload']}")
    print(f"🧠 Skip Embeddings: {config['embeddings']['skip']}")
    print(f"📦 Batch Size: {config['embeddings']['batch_size']}")
    print(f"🤖 CLIP Model: {config['embeddings']['model']}")
    print("=" * 60 + "\n")
    
    # Convert task string to enum
    task_mapping = {
        "classification": DatasetTask.CLASSIFICATION,
        "detection": DatasetTask.DETECTION,
        "segmentation": DatasetTask.SEGMENTATION,
        "pose": DatasetTask.POSE,
        "obb": DatasetTask.OBB
    }
    dataset_task = task_mapping[config['dataset']['task']]
    
    # Step 1: Prepare dataset
    print("📁 Step 1: Preparing dataset...")
    is_already_loaded, dataset = prepare_voxel_dataset(
        dataset_path=config['dataset']['path'],
        dataset_name=config['dataset']['name'],
        force_reload=config['dataset']['reload'],
        dataset_task=dataset_task,
    )
    
    if not is_already_loaded and not config['embeddings']['skip']:
        patches_field = get_box_field_from_task(task=dataset_task)
        
        # For pose estimation, we use bounding boxes to extract patches
        if dataset_task == DatasetTask.POSE:
            patches_field = get_box_field_from_task(task=DatasetTask.DETECTION)
        
        # Step 2: Load CLIP model
        print("\n🤖 Step 2: Loading CLIP model...")
        embeddings_model = foz.load_zoo_model(config['embeddings']['model'])
        print(f"✓ Loaded {config['embeddings']['model']}")
        
        # Step 3: Compute visualizations
        print("\n🧠 Step 3: Computing embeddings and visualizations...")
        compute_visualizations(
            dataset=dataset,
            model=embeddings_model,
            batch_size=config['embeddings']['batch_size'],
            patches_field=patches_field,
            dataset_task=dataset_task,
        )
        
        # Step 4: Generate thumbnails
        print("\n🖼️ Step 4: Generating thumbnails for optimized UI...")
        thumbnail_dir = os.path.join(config['thumbnail_dir'], config['dataset']['name'])
        generate_thumbnails(
            dataset=dataset,
            thumbnail_dir_path=thumbnail_dir,
        )
    else:
        if is_already_loaded:
            print("✓ Dataset already loaded, skipping processing...")
        if config['embeddings']['skip']:
            print("✓ Skipping embeddings computation as requested")
    
    # Step 5: Launch app
    if not config['no_launch']:
        print("\n🚀 Step 5: Launching FiftyOne app...")
        
        session = fo.launch_app(
            dataset,
            port=config['port']
        )
        
        print(f"\n🌐 App running at: http://localhost:{config['port']}")
        print("📊 Dataset: " + config['dataset']['name'])
        print("🎯 Task: " + config['dataset']['task'])
        print("\nPress Ctrl+C to stop the app\n")
        
        try:
            session.wait()

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            print("\n✓ App closed successfully")
            print("=" * 60)
    else:
        print("\n✅ Processing complete. Dataset saved as:", config['dataset']['name'])
        print("To launch the app later, run:")
        print(f"    fiftyone app launch {config['dataset']['name']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
