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

Clone the repository and set up your environment in just a few steps:

```bash
# Clone the repository
git clone https://github.com/picsalex/yolo-dataset-quality-analysis.git
cd yolo-dataset-quality-analysis

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Run Analysis

Configure your dataset path and run the analysis:

```python
# Edit main.py to configure your dataset
dataset_path = "/path/to/your/dataset"  # Your YOLO dataset path
dataset_name = "my_dataset"             # Name for FiftyOne dataset
dataset_task = DatasetTask.DETECTION    # Choose your task type

# Run the analysis
python main.py
```

## 📊 Supported Tasks

This tool supports all major YOLO dataset formats:

| Task | Enum Value | Description |
|------|------------|-------------|
| 🎯 **Detection** | `DatasetTask.DETECTION` | Object detection with bounding boxes |
| 🎨 **Segmentation** | `DatasetTask.SEGMENTATION` | Instance segmentation with polygons |
| 🏷️ **Classification** | `DatasetTask.CLASSIFICATION` | Image classification |
| 📐 **OBB** | `DatasetTask.OBB` | Oriented bounding boxes |
| 🤸 **Pose** | `DatasetTask.POSE` | Keypoint detection / Pose estimation |

## 📁 Dataset Format

### Detection Format
```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   └── image2.txt
│   ├── val/
│   └── test/
└── data.yaml
```

**Label format (`.txt` files):**
```
class_id x_center y_center width height
```
- All values are normalized [0, 1]
- One object per line

### Segmentation Format
Same directory structure as detection.

**Label format (`.txt` files):**
```
class_id x1 y1 x2 y2 x3 y3 ...
```
- Polygon coordinates (minimum 3 points)
- All values are normalized [0, 1]

### Classification Format
```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image3.jpg
│       └── image4.jpg
├── val/
│   ├── class1/
│   └── class2/
└── test/
    ├── class1/
    └── class2/
```
- Images organized in class subdirectories
- No separate label files needed

### OBB (Oriented Bounding Box) Format
Same directory structure as detection.

**Label format (`.txt` files):**
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```
- Four corner points of the oriented rectangle
- All values are normalized [0, 1]

### Pose/Keypoint Format
Same directory structure as detection.

**Label format (`.txt` files):**
```
class_id x_center y_center width height x1 y1 v1 x2 y2 v2 ...
```
- Bounding box (first 4 values after class_id)
- Keypoints in groups of 3: (x, y, visibility)
- Visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
- All coordinates are normalized [0, 1]

## 📂 Supported Split Names

The tool automatically recognizes the following split directory names:

| Standard Splits | COCO-style Splits | Description |
|----------------|-------------------|-------------|
| `train` | `train2017` | Training data |
| `val` | `val2017` | Validation data |
| `valid` | - | Alternative validation naming |
| `test` | `test2017` | Test data |

## ⚙️ Configuration

### data.yaml Format

For all tasks except classification, include a `data.yaml` or `dataset.yaml` file:

```yaml
# Example data.yaml
path: /path/to/dataset  # Optional: dataset root path
train: images/train     # Optional: train images path
val: images/val         # Optional: validation images path
test: images/test       # Optional: test images path

# Class names (required)
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  # ... more classes

# OR as a list
names: ['person', 'bicycle', 'car', 'motorcycle']

# Number of classes
nc: 4  # Optional but recommended
```

## 🔧 Advanced Configuration

Edit `main.py` to customize your analysis:

```python
# Analysis configuration
batch_size = 16                              # Batch size for processing
thumbnail_dir = "thumbnails/my_dataset"      # Thumbnail output directory
force_reload = True                          # Force dataset reload, useful if the dataset has changed, otherwise the cached version will be used
clip_model = "clip-vit-base32-torch"         # CLIP model for embeddings
```

The available CLIP models are listed here: https://docs.voxel51.com/model_zoo/models.html.

## 📊 Output

The analysis generates:

1. **FiftyOne Dataset**: Interactive web-based dataset explorer
2. **Thumbnails**: Optimized image thumbnails (1024px height)
3. **Embeddings**: CLIP embeddings for similarity analysis
4. **Visualizations**: UMAP projections for data distribution
5. **Quality Metrics**: Dataset statistics and quality indicators

### Viewing Results

After running the analysis, the FiftyOne app will automatically open in your browser at `http://localhost:5151`.

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
