"""Resolve data URIs to local dataset directory paths."""

import json
import os
import shutil
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm


def resolve_data_path(uri: str, dataset_dir: str, force: bool = False) -> str:
    """Resolve any data URI to a local directory path.

    Supported formats:
      - ul://username/datasets/slug  — Ultralytics Platform (downloaded to dataset_dir)
      - /path/to/data.yaml           — YOLO yaml file (resolves to parent directory)
      - /path/to/dataset/            — Local directory (returned as-is)
    """
    for resolver in _RESOLVERS:
        if resolver.can_resolve(uri):
            return resolver.resolve(uri, dataset_dir, force)

    if Path(uri).suffix in (".yaml", ".yml"):
        return str(Path(uri).parent.resolve())

    return uri


class DatasetResolver(ABC):
    """Base class for URI-based dataset resolvers."""

    @abstractmethod
    def can_resolve(self, uri: str) -> bool: ...

    @abstractmethod
    def resolve(self, uri: str, dataset_dir: str, force: bool = False) -> str: ...


class UltralyticsResolver(DatasetResolver):
    """Resolve ul://username/datasets/slug URIs from the Ultralytics Platform."""

    _API = "https://platform.ultralytics.com/api/webhooks"

    def can_resolve(self, uri: str) -> bool:
        return uri.startswith("ul://")

    def resolve(self, uri: str, dataset_dir: str, force: bool = False) -> str:
        from yolo_scout.utils.logger import logger

        parts = uri[len("ul://") :].split("/")
        if len(parts) != 3 or parts[1] != "datasets":
            raise ValueError(
                f"The provided path '{uri}' is invalid:\n"
                f"  Expected : ul://<username>/datasets/<slug>\n"
                f"  Example  : ul://john/datasets/my-dataset"
            )
        username, _, slug = parts

        api_key = os.environ.get("ULTRALYTICS_API_KEY")
        if not api_key:
            raise ValueError(
                f"The provided path '{uri}' requires to set ULTRALYTICS_API_KEY:\n"
                "  1. Get your key at https://platform.ultralytics.com/settings\n"
                "  2. Export it: export ULTRALYTICS_API_KEY=<your_key>"
            )

        dest = Path(dataset_dir) / slug
        if force and dest.exists():
            logger.info(f"Reloading dataset '{slug}', current dataset version has been removed")
            shutil.rmtree(dest)
        elif (dest / "data.yaml").exists():
            logger.info(f"Using dataset '{slug}' at '{dest}'. Set reload=True to redownload and recompute everything")
            return str(dest)

        dest.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            f"{self._API}/datasets/{username}/{slug}/export",
            method="HEAD",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            logger.info(f"Requesting export for dataset '{slug}' from the Ultralytics Platform")
            with urllib.request.urlopen(req, timeout=3600) as resp:
                ndjson_url = resp.geturl()

        except urllib.error.HTTPError as e:
            _handle_http_error(e, uri)

        from yolo_scout.utils.logger import logger

        ndjson_path = dest / f"{slug}.ndjson"
        urllib.request.urlretrieve(ndjson_url, ndjson_path)  # noqa: S310
        logger.info(f"Retrieved '{slug}.ndjson', processing and downloading images")
        _ndjson_to_yolo(ndjson_path, dest)
        ndjson_path.unlink()

        return str(dest)


_RESOLVERS: list[DatasetResolver] = [UltralyticsResolver()]


def _ndjson_to_yolo(ndjson_path: Path, dest: Path) -> None:
    """Convert an Ultralytics Platform NDJSON export to a YOLO directory structure."""
    with open(ndjson_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    if not lines or lines[0].get("type") != "dataset":
        raise ValueError("Invalid NDJSON: missing dataset header on line 1")

    class_names = lines[0].get("class_names", {})
    image_records = [r for r in lines[1:] if r.get("type") == "image"]

    def _process(record: dict) -> None:
        split = record["split"]
        filename = record["file"]
        img_path = dest / "images" / split / filename
        img_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(record["url"], img_path)  # noqa: S310

        boxes = record.get("annotations", {}).get("boxes", [])
        if boxes:
            label_path = dest / "labels" / split / f"{Path(filename).stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(" ".join(str(v) for v in box) for box in boxes))

    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(_process, r) for r in image_records]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            pass

    names = {int(k): v for k, v in class_names.items()}
    with open(dest / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "path": str(dest.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": names,
            },
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def _handle_http_error(e: urllib.error.HTTPError, uri: str) -> None:
    if e.code == 401:
        raise ValueError(f"Invalid or missing ULTRALYTICS_API_KEY (401) for '{uri}'") from e
    if e.code == 403:
        raise PermissionError(f"Access denied to dataset '{uri}' (403)") from e
    if e.code == 404:
        raise FileNotFoundError(f"Dataset not found: '{uri}' (404)") from e
    if e.code == 409:
        raise RuntimeError(f"Dataset '{uri}' is not ready yet — try again later (409)") from e
    raise RuntimeError(f"Unexpected HTTP {e.code} for '{uri}'") from e
