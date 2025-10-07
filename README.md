# 🎯 YOLO Dataset Quality Analysis

<div align="center">
  <img src="https://github.com/ultralytics/assets/raw/main/logo/Ultralytics-logomark-color.png" width="120" alt="Ultralytics Logo">
  
  **A comprehensive tool for analyzing and visualizing YOLO dataset quality using FiftyOne**
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
  [![FiftyOne](https://img.shields.io/badge/FiftyOne-Latest-orange.svg)](https://voxel51.com/fiftyone)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

---

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
python main.py --dataset-path /path/to/dataset --dataset-task detection

# Option 2: Config file only (if it contains path and task)
python main.py --config cfg/my_config.yaml

# Option 3: Config file + overrides
python main.py --config cfg/default.yaml --dataset-path /path/to/new/dataset
```

## ⚙️ Configuration

### Using Config File (Recommended for repeated use)

Create a `cfg/my_config.yaml`:
```yaml
dataset:
  path: "/path/to/your/dataset"
  task: "detection"  # detection, segmentation, classification, pose, obb
  name: "my_analysis"  # optional, auto-generated if not set
  reload: false

embeddings:
  skip: false
  model: "clip-vit-base32-torch"
  batch_size: 16
```

Then simply run:
```bash
python main.py --config cfg/my_config.yaml
```

### Command-Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--dataset-path` | Yes* | Path to YOLO dataset (*not needed if in config) |
| `--dataset-task` | Yes* | Task type: `classification`, `detection`, `segmentation`, `pose`, `obb` (*not needed if in config) |
| `--config` | No | Path to config YAML file |
| `--dataset-name` | No | Name for FiftyOne dataset (auto-generated from path if not set) |
| `--reload` | No | Force reload dataset |
| `--skip-embeddings` | No | Skip embedding computation |
| `--batch-size` | No | Batch size for embeddings |
| `--model` | No | CLIP model name |
| `--port` | No | FiftyOne app port (default: 5151) |
| `--no-launch` | No | Don't launch FiftyOne app |

## 💡 Examples

```bash
# Quick visualization without embeddings
python main.py --dataset-path /datasets/coco --dataset-task detection --skip-embeddings

# Use config file for common settings
python main.py --config cfg/coco_config.yaml

# Override config with different dataset
python main.py --config cfg/default.yaml --dataset-path /datasets/test --reload

# Process multiple datasets with same config
python main.py --config cfg/base.yaml --dataset-path /datasets/train --dataset-name train_analysis
python main.py --config cfg/base.yaml --dataset-path /datasets/val --dataset-name val_analysis

# High-performance processing
python main.py --dataset-path /large/dataset --dataset-task segmentation --batch-size 64 --model clip-vit-large-336
```

## 📊 Supported Tasks & Metadata

For each expected task format, the following metadata will be computed and available in FiftyOne:

### 🎯 Detection
**Format:** `class_id x_center y_center width height`
- `area`, `bbox_aspect_ratio`, `bbox_width`, `bbox_height`, `object_count`

### 🎨 Segmentation
**Format:** `class_id x1 y1 x2 y2 x3 y3 ...`
- `area`, `num_points`, `object_count`

### 🏷️ Classification
**Structure:** `dataset/{train,val,test}/{class_name}/`
- Class labels

### 📐 OBB
**Format:** `class_id x1 y1 x2 y2 x3 y3 x4 y4`
- `area`, `bbox_width`, `bbox_height`, `object_count`

### 🤸 Pose
**Format:** `class_id x_center y_center width height x1 y1 v1 x2 y2 v2 ...`
- `area`, `num_keypoints`, `bbox_aspect_ratio`, `bbox_width`, `bbox_height`, `object_count`

## 📁 Dataset Structure

The following YOLO dataset structure is expected for all tasks except classification:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml  # Class names
```

For classification, the structure should be:
```
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
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