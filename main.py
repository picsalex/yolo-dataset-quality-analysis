#!/usr/bin/env python3
"""
YOLO Dataset Quality Analysis Tool
Main script for analyzing YOLO datasets using FiftyOne

Author: Alexis Béduneau
License: MIT
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import fiftyone as fo
import fiftyone.zoo as foz
import yaml

from src.config import (
    get_box_field_from_task,
    get_color_palette,
    images_embeddings_field,
)
from src.dataset import prepare_voxel_dataset
from src.enum import DatasetTask
from src.images import generate_thumbnails
from src.voxel51 import compute_visualizations


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze YOLO dataset quality using FiftyOne",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration file (optional)
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration YAML file"
    )

    # Dataset arguments (conditionally required)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to YOLO dataset (required if no config file)",
    )

    parser.add_argument(
        "--dataset-task",
        type=str,
        default=None,
        choices=[task.value for task in DatasetTask],
        help="Dataset task type (required if no config file)",
    )

    # Optional arguments
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name for FiftyOne dataset (default: auto-generated from path)",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        default=None,
        help="Force reload dataset even if it exists",
    )

    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        default=None,
        help="Skip embedding computation",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for embedding computation",
    )

    parser.add_argument(
        "--model", type=str, default=None, help="CLIP model name for embeddings"
    )

    parser.add_argument(
        "--thumbnail-dir", type=str, default=None, help="Base directory for thumbnails"
    )

    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=None,
        help="Thumbnail width (width in pixels)",
    )

    parser.add_argument("--port", type=int, default=None, help="Port for FiftyOne app")

    parser.add_argument(
        "--no-launch",
        action="store_true",
        default=False,
        help="Don't launch FiftyOne app after processing",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build configuration from default.yaml, optional config file, and arguments"""

    default_config_path = "cfg/default.yaml"
    if not os.path.exists(default_config_path):
        print(f"❌ Error: default configuration file not found: {default_config_path}")
        sys.exit(1)

    config = load_config(default_config_path)

    # Load user-specified config file if provided
    if args.config:
        if not os.path.exists(args.config):
            print(f"❌ Config file not found: {args.config}")
            sys.exit(1)

        user_config = load_config(args.config)

        if "dataset" in user_config:
            config["dataset"].update(user_config["dataset"])

        if "embeddings" in user_config:
            config["embeddings"].update(user_config["embeddings"])

        if "thumbnails" in user_config:
            config["thumbnails"].update(user_config["thumbnails"])

    # Override with command-line arguments
    if args.dataset_path is not None:
        config["dataset"]["path"] = args.dataset_path

    if args.dataset_task is not None:
        config["dataset"]["task"] = args.dataset_task

    if args.dataset_name is not None:
        config["dataset"]["name"] = args.dataset_name

    if args.reload is not None:
        config["dataset"]["reload"] = args.reload

    if args.skip_embeddings is not None:
        config["embeddings"]["skip"] = args.skip_embeddings

    if args.batch_size is not None:
        config["embeddings"]["batch_size"] = args.batch_size

    if args.model is not None:
        config["embeddings"]["model"] = args.model

    if args.thumbnail_dir is not None:
        config["thumbnails"]["dir"] = args.thumbnail_dir

    if args.thumbnail_width is not None:
        config["thumbnails"]["width"] = args.thumbnail_width

    if args.port is not None:
        config["port"] = args.port

    config["no_launch"] = args.no_launch

    # Validate required fields
    if not config["dataset"].get("path"):
        print(
            "❌ Error: dataset path is required (use --dataset-path or specify in config file)"
        )
        sys.exit(1)

    if not config["dataset"].get("task"):
        print(
            "❌ Error: dataset task is required (use --dataset-task or specify in config file)"
        )
        sys.exit(1)

    # Auto-generate dataset name if not provided or if it"s still set to "default"
    if (
        not config["dataset"].get("name")
        or config["dataset"]["name"].strip() == "default"
    ):
        dataset_path = Path(config["dataset"]["path"])
        config["dataset"]["name"] = (
            dataset_path.parent.name if not dataset_path.is_dir() else dataset_path.name
        )

    return config


def main():
    args = parse_arguments()
    config = build_config(args)

    dataset_path = config["dataset"]["path"]
    dataset_task = DatasetTask(config["dataset"]["task"])
    thumbnail_width = config["thumbnails"]["width"]

    # Validate dataset path exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    # Validate task and convert to enum
    valid_tasks = [task.value for task in DatasetTask]
    if config["dataset"]["task"] not in valid_tasks:
        print(
            f"❌ Invalid task: {config['dataset']['task']}. Must be one of {valid_tasks}"
        )
        sys.exit(1)

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
    print(f"🖼️ Thumbnail size: ({config['thumbnails']['width']}, -1)")
    print("=" * 60 + "\n")

    # Step 1: Prepare dataset
    print("📁 Step 1: Preparing dataset...")
    is_already_loaded, dataset = prepare_voxel_dataset(
        dataset_path=config["dataset"]["path"],
        dataset_name=config["dataset"]["name"],
        force_reload=config["dataset"]["reload"],
        thumbnail_width=config["thumbnails"]["width"],
        dataset_task=dataset_task,
    )

    if is_already_loaded:
        print(
            f"Dataset '{config['dataset']['name']}' is already loaded, skipping preparation."
        )

    elif not config["embeddings"]["skip"]:
        patches_field = get_box_field_from_task(task=dataset_task)

        # For pose estimation, we use bounding boxes to extract patches
        if dataset_task == DatasetTask.POSE:
            patches_field = get_box_field_from_task(task=DatasetTask.DETECTION)

        # Step 2: Load CLIP model
        print("\n🤖 Step 2: Loading CLIP model...")
        embeddings_model = foz.load_zoo_model(config["embeddings"]["model"])
        print(f"Loaded {config['embeddings']['model']}")

        # Step 3: Compute visualizations
        print("\n🧠 Step 3: Computing embeddings and visualizations...")
        compute_visualizations(
            dataset=dataset,
            model=embeddings_model,
            dataset_task=dataset_task,
            batch_size=config["embeddings"]["batch_size"],
            patches_embeddings_brain_key=patches_field,
            images_embeddings_brain_key=images_embeddings_field,
        )

    else:
        print("Skipping embeddings computation requested by user.")

    # Step 4: Generate thumbnails
    if thumbnail_width > 0:
        if (
            "thumbnail_width" in dataset.info
            and dataset.info["thumbnail_width"] == thumbnail_width
            and is_already_loaded
        ):
            print(
                f"\n🖼️ Step 4: Thumbnails of size ({thumbnail_width}, -1) already exist, skipping generation..."
            )

        else:
            print(
                f"\n🖼️ Step 4: Generating thumbnails of size ({thumbnail_width}, -1) for optimized Fiftyone dashboard..."
            )
            thumbnail_dir = Path(
                os.path.join(
                    config.get("thumbnail_dir", "thumbnails"), config["dataset"]["name"]
                )
            ).resolve()
            generate_thumbnails(
                dataset=dataset,
                thumbnail_dir_path=str(thumbnail_dir),
                thumbnail_width=config["thumbnails"]["width"],
            )
    else:
        print(
            "\n🖼️ Step 4: Skipping thumbnail generation as the provided width is not greater than 0"
        )

    # Launch FiftyOne app unless no_launch is specified
    if not config.get("no_launch", False):
        print("\n🚀 Launching FiftyOne app:")

        if "class_names" not in dataset.info:
            raise ValueError(
                "Dataset class names not found in dataset info. Cannot launch app with color scheme. Please force reload the dataset."
            )

        color_palette = get_color_palette(labels=dataset.info["class_names"])
        field_name = get_box_field_from_task(task=dataset_task)

        # For pose estimation, we only color the bounding boxes, keypoints colors are picked randomly
        if dataset_task == DatasetTask.POSE:
            field_name = get_box_field_from_task(task=DatasetTask.DETECTION)

        _ = fo.launch_app(
            dataset,
            port=config.get("port", 5151),
            color_scheme=fo.ColorScheme(
                color_by="value",
                fields=[
                    {
                        "path": field_name,
                        "valueColors": color_palette,
                    }
                ],
                multicolor_keypoints=True,
            ),
        )

        print(f"\n🌐 App running at: http://localhost:{config.get('port', 5151)}")
        print("📊 Dataset: " + config["dataset"]["name"])
        print("🎯 Task: " + config["dataset"]["task"])
        print("\nTo exit, close the App or press ctrl + c")

        try:
            while True:
                import time

                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\nShutting down gracefully...")

        finally:
            print("App closed successfully")
            print("=" * 60)

    else:
        print("\n✅ Processing complete. Dataset saved as:", config["dataset"]["name"])
        print("To launch the app later, run:")
        print(f"    fiftyone app launch {config['dataset']['name']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
