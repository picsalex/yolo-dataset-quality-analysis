---
tags:
  - Guide
  - Setup
---

# Installation

## Requirements

- Python 3.10 or later
- A CUDA-capable GPU (recommended for embedding computation)

## Install

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

## Key dependencies

YoloScout depends on several heavyweight libraries that are installed automatically:

| Package              | Purpose                            |
| -------------------- | ---------------------------------- |
| `torch`              | Deep learning framework            |
| `fiftyone`           | Dataset visualization platform     |
| `open_clip_torch`    | CLIP embedding models              |
| `transformers`       | HuggingFace model loading          |
| `umap-learn`         | Dimensionality reduction           |
| `shapely`            | Polygon IoU computation            |
| `opencv-python-headless` | Image processing              |

!!! note "First run"

    The first time you use a CLIP model, it will be downloaded automatically. Subsequent runs use the cached model.

## Verify installation

```bash
yolo-scout --help
```

!!! success "Expected output"

    You should see usage information with all available configuration options: `data`, `task`, `model`, `batch`, and more.
