"""Configuration management for the application."""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from src.core.enums import DatasetTask, EmbeddingsModel
from src.utils.logger import logger


@dataclass
class Config:
    """Application configuration."""

    # Dataset
    dataset_path: str
    dataset_name: str
    dataset_task: DatasetTask
    force_reload: bool

    # Embeddings
    skip_embeddings: bool
    embeddings_model: EmbeddingsModel
    batch_size: int
    mask_background: bool

    # Thumbnails
    thumbnail_width: int
    thumbnail_dir: str

    # App
    port: int
    skip_launch: bool

    @classmethod
    def from_cli(cls) -> "Config":
        """Build configuration from CLI args, YAML files, and defaults."""
        args = _parse_arguments()
        config_dict = _build_config_dict(args)
        return cls._from_dict(config_dict)

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config instance from dictionary."""
        return cls(
            dataset_path=config_dict["dataset"]["path"],
            dataset_name=config_dict["dataset"]["name"],
            dataset_task=DatasetTask(config_dict["dataset"]["task"]),
            force_reload=config_dict["dataset"]["reload"],
            skip_embeddings=config_dict["embeddings"]["skip"],
            embeddings_model=EmbeddingsModel(config_dict["embeddings"]["model"]),
            batch_size=config_dict["embeddings"]["batch_size"],
            mask_background=config_dict["embeddings"]["mask_background"],
            thumbnail_width=config_dict["thumbnails"]["width"],
            thumbnail_dir=config_dict["thumbnails"]["dir"],
            port=config_dict.get("port", 5151),
            skip_launch=config_dict.get("skip_launch", False),
        )

    def validate(self):
        """Validate configuration values."""
        if not os.path.exists(self.dataset_path):
            logger.error(f"Dataset path does not exist: {self.dataset_path}")
            sys.exit(1)

    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get model kwargs for embeddings model."""
        return self.embeddings_model.get_model_kwargs()


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
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
        help="Dataset task type (required if no config file)",
    )

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
        "--embeddings-model",
        type=str,
        default=None,
        help="CLIP model name for embeddings",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for embedding computation",
    )

    parser.add_argument(
        "--mask-background",
        action="store_true",
        default=None,
        help="Enable background masking in patch crops for segmentation/OBB tasks (disabled by default)",
    )

    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=None,
        help="Thumbnail width in pixels (height is scaled proportionally)",
    )

    parser.add_argument(
        "--thumbnail-dir", type=str, default=None, help="Base directory for thumbnails"
    )

    parser.add_argument("--port", type=int, default=None, help="Port for FiftyOne app")

    parser.add_argument(
        "--skip-launch",
        action="store_true",
        default=False,
        help="Skip launching the FiftyOne app after processing",
    )

    return parser.parse_args()


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration file '{config_path}': {e}")
        raise


def _build_config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Build configuration dictionary from default.yaml, optional config file, and arguments."""
    default_config_path = "cfg/default.yaml"
    if not os.path.exists(default_config_path):
        logger.error(f"Default configuration file not found: {default_config_path}")
        sys.exit(1)

    config = _load_yaml_config(default_config_path)

    # Load user-specified config file if provided
    if args.config:
        if not os.path.exists(args.config):
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)

        user_config = _load_yaml_config(args.config)

        if "dataset" in user_config:
            config["dataset"].update(user_config["dataset"])

        if "embeddings" in user_config:
            config["embeddings"].update(user_config["embeddings"])

        if "thumbnails" in user_config:
            config["thumbnails"].update(user_config["thumbnails"])

        if "port" in user_config:
            config["port"] = user_config["port"]

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

    if args.embeddings_model is not None:
        config["embeddings"]["model"] = args.embeddings_model

    if args.batch_size is not None:
        config["embeddings"]["batch_size"] = args.batch_size

    if args.mask_background is not None:
        config["embeddings"]["mask_background"] = True

    if args.thumbnail_dir is not None:
        config["thumbnails"]["dir"] = args.thumbnail_dir

    if args.thumbnail_width is not None:
        config["thumbnails"]["width"] = args.thumbnail_width

    if args.port is not None:
        config["port"] = args.port

    # Defaults to False if not specified
    config["skip_launch"] = args.skip_launch

    # Validate the dataset path
    if not config["dataset"].get("path"):
        logger.error(
            "Dataset path is required (use --dataset-path or specify in config file)"
        )
        sys.exit(1)

    elif not os.path.exists(config["dataset"]["path"]):
        logger.error(f"Dataset path does not exist: {config['dataset']['path']}")
        sys.exit(1)

    # Validate the dataset task
    if not config["dataset"].get("task"):
        logger.error(
            "Dataset task is required (use --dataset-task or specify in config file)"
        )
        sys.exit(1)

    elif not DatasetTask.is_valid_value(value=config["dataset"]["task"]):
        logger.error(
            f"Dataset task '{config['dataset']['task']}' not supported, possible values are: {[t.value for t in DatasetTask]}"
        )
        sys.exit(1)

    # Auto-generate dataset name if not provided or if it's still set to "default"
    if (
        not config["dataset"].get("name")
        or config["dataset"]["name"].strip() == "default"
    ):
        dataset_path = Path(config["dataset"]["path"])
        config["dataset"]["name"] = (
            dataset_path.parent.name if not dataset_path.is_dir() else dataset_path.name
        )

    # Validate embeddings model
    if not config["embeddings"].get("model") or not EmbeddingsModel.is_valid_value(
        value=config["embeddings"]["model"]
    ):
        logger.warning(
            f"Embeddings model '{config['embeddings']['model']}' not supported, possible values are: {[e.value for e in EmbeddingsModel]}. Defaulting to 'openai_clip'.\n"
        )
        config["embeddings"]["model"] = EmbeddingsModel.OPENAI_CLIP.value

    return config
