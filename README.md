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
# Minimal usage - only 2 required arguments
python main.py --dataset-path /path/to/your/dataset --dataset-task detection
```

## ⚙️ Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset-path` | **Yes** | - | Path to YOLO dataset |
| `--dataset-task` | **Yes** | - | Task type: `classification`, `detection`, `segmentation`, `pose`, `obb` |
| `--dataset-name` | No | Auto from path | Name for FiftyOne dataset |
| `--config` | No | None | Optional config YAML file |
| `--reload` | No | False | Force reload dataset |
| `--skip-embeddings` | No | False | Skip embedding computation |
| `--batch-size` | No | 16 | Batch size for embeddings |
| `--model` | No | clip-vit-base32-torch | CLIP model name |
| `--thumbnail-dir` | No | thumbnails | Directory for thumbnails |
| `--port` | No | 5151 | FiftyOne app port |
| `--no-launch` | No | False | Don't launch FiftyOne app |

## 📝 Configuration (Optional)

You can use a YAML config file for default values:

```yaml
# cfg/default.yaml
dataset:
  path: "/path/to/your/yolo/dataset"
  name: "yolo_dataset"
  task: "detection"
  reload: false

embeddings:
  skip: false
  model: "clip-vit-base32-torch"
  batch_size: 16
```

Usage with config:
```bash
# Config file + required arguments
python main.py --config cfg/default.yaml --dataset-path /path/to/dataset --dataset-task detection

# Override config values with arguments
python main.py --config cfg/default.yaml --dataset-path /path/to/dataset --dataset-task detection --batch-size 32
```

## 📊 Supported Tasks & Metadata

### 🎯 Detection Task
**Label Format:** `class_id x_center y_center width height`

**Metadata:**
- `area`: Bounding box area in pixels²
- `bbox_aspect_ratio`: Width/height ratio
- `bbox_width`, `bbox_height`: Dimensions in pixels
- `object_count`: Objects per image

### 🎨 Segmentation Task
**Label Format:** `class_id x1 y1 x2 y2 x3 y3 ...`

**Metadata:**
- `area`: Polygon area in pixels² (Shapely)
- `num_points`: Number of polygon vertices
- `object_count`: Segments per image

### 🏷️ Classification Task
**Structure:** `dataset/{train,val,test}/{class_name}/images.jpg`

**Metadata:**
- Image classification labels
- Class distribution statistics

### 📐 OBB Task
**Label Format:** `class_id x1 y1 x2 y2 x3 y3 x4 y4`

**Metadata:**
- `area`: OBB area in pixels²
- `bbox_width`, `bbox_height`: OBB dimensions
- `object_count`: OBBs per image

### 🤸 Pose Task
**Label Format:** `class_id x_center y_center width height x1 y1 v1 x2 y2 v2 ...`

**Metadata:**
- `area`: Bounding box area
- `num_keypoints`: Detected keypoints per instance
- `bbox_aspect_ratio`, `bbox_width`, `bbox_height`: Box dimensions
- `object_count`: Pose instances per image

## 💡 Examples

```bash
# Quick visualization without embeddings
python main.py --dataset-path /datasets/coco --dataset-task detection --skip-embeddings

# High-performance with larger batch
python main.py --dataset-path /datasets/large --dataset-task segmentation --batch-size 64

# Process without launching app
python main.py --dataset-path /datasets/test --dataset-task detection --no-launch

# Custom model
python main.py --dataset-path /datasets/data --dataset-task pose --model clip-vit-large-336
```

## 📁 Dataset Structure

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

## 🐛 Troubleshooting

**Dataset not found:**
```bash
# Check path exists and has correct structure
ls -la /path/to/dataset/
```

**Memory issues:**
```bash
# Reduce batch size
python main.py --dataset-path /path --dataset-task detection --batch-size 8
```

**Quick preview:**
```bash
# Skip heavy processing
python main.py --dataset-path /path --dataset-task detection --skip-embeddings
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.
