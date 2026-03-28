---
tags:
  - Guide
---

<div align="center" markdown>

<picture>
  <img src="https://raw.githubusercontent.com/picsalex/yolo-scout/main/images/yolo-scout-black.png" alt="YoloScout" width="500">
</picture>

# YoloScout

A comprehensive YOLO dataset quality analysis tool powered by
[FiftyOne](https://docs.voxel51.com/). Load, analyze, and visualize YOLO
datasets with CLIP embeddings, quality metrics, and an interactive web UI.

[![PyPI](https://img.shields.io/pypi/v/yolo-scout.svg)](https://pypi.org/project/yolo-scout/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FiftyOne](https://img.shields.io/badge/FiftyOne-Latest-orange.svg)](https://voxel51.com/fiftyone)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/picsalex/yolo-scout/blob/main/LICENSE)

</div>

---

<div align="center" markdown>

![FiftyOne UI Screenshot with OBB dataset](https://raw.githubusercontent.com/picsalex/yolo-scout/main/images/voxel_ui.avif){ width="100%" }

</div>

## Features

- :material-magnify: **Dataset analysis** — load any YOLO-formatted dataset and
  inspect annotations, metadata, and quality
- :material-brain: **CLIP embeddings** — compute image and patch-level
  embeddings with [OpenAI CLIP](https://github.com/openai/CLIP),
  [MetaCLIP](https://github.com/facebookresearch/MetaCLIP), or
  [SigLIP](https://github.com/google-research/big_vision)
- :material-chart-scatter-plot: **UMAP visualization** — explore embedding space
  with interactive 2D projections
- :material-image-check: **Quality metrics** — blurriness, brightness, entropy,
  and aspect ratio per image and patch
- :material-layers: **Multi-task support** — detect, classify, segment, pose,
  and oriented bounding boxes
- :material-eye: **Interactive UI** — browse your dataset in the FiftyOne web app
  with custom color schemes and the built-in
  [image-adjuster plugin](#bundled-plugins)
- :material-cloud-download: **Multiple data sources** — local directories, YAML
  files, NDJSON exports, and Ultralytics Platform URLs
- :material-cached: **Smart caching** — persistent FiftyOne datasets avoid
  redundant processing

## Quick start

!!! example "Get running in under a minute"

    === "pip"

        ```bash
        pip install yolo-scout
        ```

    === "uv"

        ```bash
        uv add yolo-scout
        ```

    === "From source"

        ```bash
        git clone https://github.com/picsalex/yolo-scout.git
        cd yolo-scout
        uv sync
        ```

    Then analyze a dataset:

    ```bash
    # Object detection dataset
    yolo-scout data=/path/to/dataset task=detect

    # Segmentation with MetaCLIP embeddings
    yolo-scout data=/path/to/data.yaml task=segment model=metaclip_400m

    # Ultralytics Platform dataset
    ULTRALYTICS_API_KEY=<your_key> yolo-scout data=ul://username/datasets/my-dataset task=detect

    # Config file with CLI overrides
    yolo-scout config=my_config.yaml batch=8

    # Force reload of an existing dataset
    yolo-scout data=/path/to/dataset task=detect reload=true

    # Skip embeddings for a quick overview
    yolo-scout data=/path/to/dataset task=detect skip_embeddings=true
    ```

## How it works

```
CLI Input                     Pipeline                        Output
─────────                     ────────                        ──────
data=/path task=detect ──────▶ Validate config
                               Load dataset into FiftyOne
                               Compute CLIP embeddings
                               Compute quality metrics    ───▶ FiftyOne Web App
                               Generate thumbnails             (interactive UI)
                               Launch visualization
```

1. **Validate** — parse CLI args, merge with config file and defaults
2. **Load** — discover splits, parse YOLO annotations, create FiftyOne dataset
3. **Embed** — compute CLIP image and patch embeddings, run UMAP
4. **Analyze** — calculate quality metrics (blurriness, brightness, entropy)
5. **Visualize** — launch FiftyOne app with color-coded annotations

!!! tip "Caching"

    Datasets are persisted in FiftyOne's database. Re-running the same dataset
    skips loading and goes straight to visualization. Use `reload=true` to force
    reprocessing.

## Bundled plugins

YoloScout ships with a custom FiftyOne plugin that is automatically installed
at startup — no manual setup required.

| Plugin | Description | How to use |
| ------ | ----------- | ---------- |
| `@ultralytics/image-adjuster` | Adjust image brightness, contrast, and label overlay opacity | Open a sample, then click the slider icon in the bottom-left corner |

## Acknowledgments

- Built with [FiftyOne](https://voxel51.com/fiftyone) by Voxel51
- Inspired by the [Ultralytics](https://ultralytics.com) YOLO ecosystem
- CLIP models from [OpenAI](https://openai.com/research/clip),
  [Meta](https://github.com/facebookresearch/MetaCLIP), and
  [Google](https://github.com/google-research/big_vision)
