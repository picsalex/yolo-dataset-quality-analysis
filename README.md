# 🎯 Ultralytics YOLO Dataset Quality Analysis tool

<div align="center">
  <img src="https://github.com/ultralytics/assets/raw/main/logo/Ultralytics-logomark-color.png" width="120" alt="Ultralytics Logo">

  **A comprehensive tool for analyzing and visualizing YOLO dataset quality using FiftyOne**

  [![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
  [![FiftyOne](https://img.shields.io/badge/FiftyOne-Latest-orange.svg)](https://voxel51.com/fiftyone)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

---

<div align="center">
  <img src="images/voxel_ui.avif" alt="FiftyOne UI Screenshot with OBB dataset" width="100%">
</div>


## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/picsalex/yolo-dataset-quality-analysis.git
cd yolo-dataset-quality-analysis

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Basic Usage

```bash
# Option 1: Command-line only (no config file)
python main.py --dataset-path /path/to/dataset --dataset-task detect

# Option 2: Config file only (more details below)
python main.py --config cfg/my_config.yaml

# Option 3: Config file + overrides
python main.py --config cfg/default.yaml --batch_size 8

# Option 4: Force reload of an existing dataset
python main.py --dataset-path /path/to/dataset --dataset-task detect --reload
```

If you want to use the configuration file option, you can either override the default config file located at `cfg/default.yaml` or create your own config file (e.g., `cfg/my_config.yaml`) with the following structure:

```yaml
dataset:
  path: "/path/to/your/dataset"
  name: "my_dataset"  # optional, auto-generated if not set
  task: "detect"  # detect, segment, classify, pose, obb
  reload: false

embeddings:
  skip: false
  model: "openai_clip"
  dir: "./models/fiftyone"
  batch_size: 16
  mask_background: true  # Mask background for segment/OBB patch crops

thumbnails:
  dir: "./thumbnails"
  width: 800
```

### Command-Line Arguments

| Argument                  | Type   | Default               | Description                                                                                                                                                              |
|---------------------------|--------|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--dataset-path`          | `str`  | `None`                | Path to your dataset. Required unless provided in config file and must follow the [YOLO format](https://docs.ultralytics.com/datasets/).                                 |
| `--dataset-task`          | `str`  | `'detect'`            | Task type: `classify`, `detect`, `segment`, `pose`, `obb`. Required unless in config. More info on the tasks [below](#-supported-tasks-and-image-metadata).              |
| `--dataset-name`          | `str`  | `'default'`           | Name for the FiftyOne dataset. Auto-generated from path if not set.                                                                                                      |
| `--config`                | `str`  | `None`                | Path to config YAML file. Overrides default settings.                                                                                                                    |
| `--reload`                | `bool` | `false`               | Force reload of the dataset even if it already exists. The current dataset will be deleted and recreated.                                                                |
| `--skip-embeddings`       | `bool` | `false`               | Skip CLIP embedding computation (useful for quick visualization).                                                                                                        |
| `--batch-size`            | `int`  | `16`                  | Batch size used during CLIP embedding computation.                                                                                                                       |
| `--no-mask-background`    | `bool` | `false`               | Disable background masking for patch crops in segmentation/OBB tasks. Masking is enabled by default, replacing background with gray (114, 114, 114).                     |
| `--model`                 | `str`  | `'openai_clip'`       | Embeddings model to use for embedding computation. Possible values are `openai_clip`, `metaclip_400m`, `metaclip_fullcc` and `siglip_base_224`.                          |
| `--embeddings-models-dir` | `str`  | `'./models/fiftyone'` | Path to the directory where the embeddings models are saved.                                                                                                             |
| `--thumbnail-dir`         | `str`  | `'./thumbnails'`      | Path to the directory where the thumbnails are saved.                                                                                                                    |
| `--thumbnail-width`       | `int`  | `800`                 | Width (in pixels) of the generated image thumbnails in FiftyOne. The height is adjusted automatically to maintain aspect ratio. Set to `-1` to disable thumbnail saving. |
| `--port`                  | `int`  | `5151`                | Port to launch the FiftyOne app on.                                                                                                                                      |
| `--no-launch`             | `bool` | `false`               | Prevents launching the FiftyOne app in the browser.                                                                                                                      |

## 📊 Supported tasks and image metadata

For each expected task format, the following metadata will be computed and available in FiftyOne for each annotation:

| Task                                                       | Available parameters when using the UI                                   |
|------------------------------------------------------------|--------------------------------------------------------------------------|
| [`classify`](https://docs.ultralytics.com/tasks/classify/) | `cls_label.label`                                                        |
| [`detect`](https://docs.ultralytics.com/tasks/detect/)     | `area`, `aspect_ratio`, `width`, `height`, `iou_score`                   |
| [`segment`](https://docs.ultralytics.com/tasks/segment/)   | `area`, `num_keypoints`, `width`, `height`, `iou_score`                  |
| [`obb`](https://docs.ultralytics.com/tasks/obb/)           | `area`, `width`, `height`, `iou_score`                                   |
| [`pose`](https://docs.ultralytics.com/tasks/pose/)         | `area`, `num_keypoints`, `aspect_ratio`, `width`, `height`, `iou_score`  |

Also, for each image, the following metadata will be computed:

| Image Metadata          | Description                                             |
|-------------------------|---------------------------------------------------------|
| `oject_count`           | Number of objects in the image                          |
| `metadata.size_bytes`   | Size of the image file in bytes                         |
| `metadata.width`        | Width of the image in pixels                            |
| `metadata.height`       | Height of the image in pixels                           |
| `metadata.mime_type`    | MIME type of the image (e.g., `image/jpeg`)             |
| `metadata.num_channels` | Number of color channels (e.g., 3 for RGB)              |

## ⭐️ Supported Models

All models use **224x224 input resolution**. This is a constraint imposed by FiftyOne's OpenCLIP integration - higher resolution variants (384, 512) cause preprocessing errors when computing embeddings. The 224x224 resolution provides excellent quality for most computer vision tasks while maintaining compatibility with FiftyOne's model zoo.

| Model               | Description                                                                                                                                                                               | Training Dataset                                         |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| **openai_clip**     | Original OpenAI CLIP model with ViT-B/32 architecture. Hosted on GitHub releases for offline usage. This is the default model and works without internet connection after first download. | [OpenAI CLIP](https://github.com/openai/CLIP)            |
| **metaclip_400m**   | MetaCLIP model trained on curated 400M image-text pairs. Offers improved data quality and better embeddings compared to OpenAI CLIP while maintaining the same speed and architecture.    | [MetaCLIP](https://github.com/facebookresearch/MetaCLIP) |
| **metaclip_fullcc** | MetaCLIP model trained on the full CommonCrawl dataset. Provides the highest quality embeddings among MetaCLIP variants with more diverse training data.                                  | [MetaCLIP](https://github.com/facebookresearch/MetaCLIP) |
| **siglip_base_224** | SigLIP (Sigmoid Loss for Language-Image Pre-training) base model. Uses improved sigmoid loss function for better performance with smaller batch sizes and more efficient training.        | [SigLIP](https://github.com/google-research/big_vision)  |

### Model Selection Guide

- **Use `openai_clip`** if you want to use the most common embeddings model
- **Use `metaclip_400m`** for better quality embeddings (recommended default)
- **Use `metaclip_fullcc`** when you need the highest quality embeddings
- **Use `siglip_base_224`** as an alternative to CLIP-based models

All models have similar inference speed and produce 512-dimensional embeddings with full support for FiftyOne visualization and analysis features.

## ⚒️ Dataset Structure

This tool supports two common YOLO dataset directory structures:

### Format 1: Type-First Structure
```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── val/
│   │   ├── image1.jpg
│   │   └── ...
│   └── test/
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    ├── val/
    │   ├── image1.txt
    │   └── ...
    └── test/
        └── ...
```

In this format, images and labels are organized by type first, then by split.

### Format 2: Split-First Structure
```
dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── val/
│   ├── images/
│   │   └── ...
│   └── labels/
│       └── ...
└── test/
    ├── images/
    │   └── ...
    └── labels/
        └── ...
```

In this format, the dataset is organized by split first, then by type (images/labels).

## ⌨️ FiftyOne commands

If you have used this tool at least one time to visualize a dataset, you can then use the following commands bellow to interact with the FiftyOne datasets and application:

```bash
# List all the datasets
fiftyone datasets list

# Delete a specific dataset using its name
fiftyone datasets delete <dataset_name>

# Delete all datasets
python -c "import fiftyone as fo; [fo.delete_dataset(name) for name in fo.list_datasets()]"

# Launch the FiftyOne app
fiftyone app launch

# Launch the FiftyOne app and pre-select a dataset using its name
fiftyone app launch <dataset_name>
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FiftyOne](https://voxel51.com/fiftyone) by Voxel51
- Inspired by [Ultralytics](https://ultralytics.com) YOLO ecosystem
- CLIP models from [OpenAI](https://openai.com/research/clip)

---

<div align="center">
  Made with ❤️ for the YOLO community
</div>
