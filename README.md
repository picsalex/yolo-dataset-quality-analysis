# 🎯 YOLO Dataset Quality Analysis

<div align="center">
  <img src="https://github.com/ultralytics/assets/raw/main/logo/Ultralytics-logomark-color.png" width="120" alt="Ultralytics Logo">
  
  **A comprehensive tool for analyzing and visualizing YOLO dataset quality using FiftyOne**
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
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
python main.py --dataset-path /path/to/dataset --dataset-task detection

# Option 2: Config file only (more details below)
python main.py --config cfg/my_config.yaml

# Option 3: Config file + overrides
python main.py --config cfg/default.yaml --batch_size 8

# Option 4: Force reload of an existing dataset
python main.py --dataset-path /path/to/dataset --dataset-task detection --reload
```

If you want to use the configuration file option, you can either override the default config file located at `cfg/default.yaml` or create your own config file (e.g., `cfg/my_config.yaml`) with the following structure:

```yaml
dataset:
  path: "/path/to/your/dataset"
  task: "detect"  # detect, segment, classify, pose, obb
  name: "my_analysis"  # optional, auto-generated if not set
  reload: false

embeddings:
  skip: false
  model: "clip-vit-base32-torch"
  batch_size: 16
```

### Command-Line Arguments

| Argument            | Type   | Default                   | Description                                                                                                                                                              |
|---------------------|--------|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--dataset-path`    | `str`  | `None`                    | Path to your dataset. Required unless provided in config file and must follow the [YOLO format](https://docs.ultralytics.com/datasets/).                                 |
| `--dataset-task`    | `str`  | `"detect"`                | Task type: `classify`, `detect`, `segment`, `pose`, `obb`. Required unless in config. More info on the tasks [below](#-supported-tasks-and-image-metadata).              |
| `--dataset-name`    | `str`  | `"default"`               | Name for the FiftyOne dataset. Auto-generated from path if not set.                                                                                                      |
| `--config`          | `str`  | `None`                    | Path to config YAML file. Overrides default settings.                                                                                                                    |
| `--reload`          | `bool` | `false`                   | Force reload of the dataset even if it already exists. The current dataset will be deleted and recreated.                                                                |
| `--skip-embeddings` | `bool` | `false`                   | Skip CLIP embedding computation (useful for quick visualization).                                                                                                        |
| `--batch-size`      | `int`  | `16`                      | Batch size used during CLIP embedding computation.                                                                                                                       |
| `--model`           | `str`  | `"clip-vit-base32-torch"` | CLIP model name to use for embedding computation. The list of possible models can be found on [Voxel51's zoo](https://docs.voxel51.com/model_zoo/models.html).           |
| `--thumbnail-dir`   | `str`  | `./thumbnails`            | Path to the directory where the thumbnails are saved.                                                                                                                    |
| `--thumbnail-width` | `int`  | `800`                     | Width (in pixels) of the generated image thumbnails in FiftyOne. The height is adjusted automatically to maintain aspect ratio. Set to `-1` to disable thumbnail saving. |
| `--port`            | `int`  | `5151`                    | Port to launch the FiftyOne app on.                                                                                                                                      |
| `--no-launch`       | `bool` | `false`                   | Prevents launching the FiftyOne app in the browser.                                                                                                                      |

## 📊 Supported tasks and image metadata

For each expected task format, the following metadata will be computed and available in FiftyOne for each annotation:

| Task                                                       | Available Parameters                                       |
|------------------------------------------------------------|------------------------------------------------------------|
| [`classify`](https://docs.ultralytics.com/tasks/classify/) | `cls_label.label`                                          |
| [`detect`](https://docs.ultralytics.com/tasks/detect/)     | `area`, `aspect_ratio`, `width`, `height`                  |
| [`segment`](https://docs.ultralytics.com/tasks/segment/)   | `area`, `num_points`, `width`, `height`                    |
| [`obb`](https://docs.ultralytics.com/tasks/obb/)           | `area`, `width`, `height`                                  |
| [`pose`](https://docs.ultralytics.com/tasks/pose/)         | `area`, `num_keypoints`, `aspect_ratio`, `width`, `height` |

Also, for each image, the following metadata will be computed:

| Image Metadata          | Description                                             |
|-------------------------|---------------------------------------------------------|
| `oject_count`           | Number of objects in the image                          |
| `metadata.size_bytes`   | Size of the image file in bytes                         |
| `metadata.width`        | Width of the image in pixels                            |
| `metadata.height`       | Height of the image in pixels                           |
| `metadata.mime_type`    | MIME type of the image (e.g., `image/jpeg`)             |
| `metadata.num_channels` | Number of color channels (e.g., 3 for RGB)              |

## ⌨️ FiftyOne commands

You can use the following commands bellow to interact with the FiftyOne datasets and application:

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