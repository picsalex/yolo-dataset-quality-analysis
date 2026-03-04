"""Plugin management for YOLO Dataset Quality Analysis Tool."""

import shutil
from pathlib import Path

import fiftyone as fo
import yaml

from src.utils.logger import logger

_PLUGIN_SRC = Path(__file__).parents[2] / "plugins" / "image-adjuster"
_PLUGIN_DST_NAME = "image-adjuster"


def ensure_plugins() -> None:
    """Copy the local plugin into FiftyOne's plugin directory."""
    src = _PLUGIN_SRC.resolve()
    dst = Path(fo.config.plugins_dir).expanduser().resolve() / _PLUGIN_DST_NAME
    dst.parent.mkdir(parents=True, exist_ok=True)

    src_version = yaml.safe_load((src / "fiftyone.yml").read_text()).get("version")
    dst_yml = dst / "fiftyone.yml"
    if dst.is_symlink():
        dst.unlink()
    elif dst_yml.exists():
        dst_version = yaml.safe_load(dst_yml.read_text()).get("version")
        if dst_version == src_version:
            logger.debug("🔌 @ultralytics/image-adjuster already installed — skipping")
            return
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    logger.debug(f"🔌 @ultralytics/image-adjuster v{src_version} installed")
