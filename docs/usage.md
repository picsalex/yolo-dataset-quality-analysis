---
tags:
  - Reference
  - CLI
---

# Usage

## Basic syntax

```bash
yolo-scout [key=value ...]
```

All options are passed as `key=value` pairs. Required arguments are `data` and
`task`.

## Quick examples

!!! example "Common workflows"

    === "Local directory"

        ```bash
        yolo-scout data=/path/to/dataset task=detect
        ```

    === "data.yaml file"

        ```bash
        yolo-scout data=/path/to/data.yaml task=segment
        ```

    === "Ultralytics Platform"

        ```bash
        ULTRALYTICS_API_KEY=<your_key> yolo-scout data=ul://username/datasets/my-dataset task=detect
        ```

    === "Config file"

        ```bash
        yolo-scout config=my_config.yaml
        ```

    === "Config + overrides"

        ```bash
        yolo-scout config=default.yaml batch=8
        ```

    === "Force reload"

        ```bash
        yolo-scout data=/path/to/dataset task=detect reload=true
        ```

## All options

### Dataset

| Option        | Type   | Default                 | Description                                                                                                        |
| ------------- | ------ | ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `data`        | `str`  | *(required)*            | Path to your dataset. See [Data sources](#data-sources) for all accepted formats.                                  |
| `task`        | `str`  | `detect`                | Task type: `detect`, `classify`, `segment`, `pose`, `obb`. See [Supported tasks](tasks.md) for details.            |
| `name`        | `str`  | *(auto from path)*      | Name for the FiftyOne dataset. Auto-generated from the directory name if not set.                                  |
| `reload`      | `bool` | `false`                 | Force reload of the dataset even if it already exists. The current dataset will be deleted and recreated.           |
| `dataset_dir` | `str`  | `yolo_scout/datasets`   | Destination directory for datasets downloaded from a URL. Only used when `data` is a URL.                          |

### Embeddings

| Option            | Type   | Default       | Description                                                                                                                   |
| ----------------- | ------ | ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `skip_embeddings` | `bool` | `false`       | Skip CLIP embedding computation (useful for quick visualization).                                                             |
| `model`           | `str`  | `openai_clip` | Embeddings model to use. See [Embeddings](embeddings.md) for options and a selection guide.                                   |
| `batch`           | `int`  | `16`          | Batch size used during CLIP embedding computation.                                                                            |
| `mask_background` | `bool` | `true`        | Mask background in patch crops for segmentation/OBB tasks. When enabled, background is replaced with gray `(114, 114, 114)`.  |

### Quality

| Option         | Type   | Default | Description                                                                                        |
| -------------- | ------ | ------- | -------------------------------------------------------------------------------------------------- |
| `skip_quality` | `bool` | `false` | Skip image quality metrics computation (blurriness, brightness, aspect_ratio, entropy).            |

### Thumbnails

| Option            | Type   | Default                  | Description                                                                                                       |
| ----------------- | ------ | ------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `thumbnail_dir`   | `str`  | `yolo_scout/thumbnails`  | Path to the directory where thumbnails are saved.                                                                 |
| `thumbnail_width` | `int`  | `800`                    | Width (in pixels) of generated thumbnails. Height is adjusted automatically to maintain aspect ratio. Set to `-1` to disable. |

### App

| Option        | Type   | Default | Description                              |
| ------------- | ------ | ------- | ---------------------------------------- |
| `port`        | `int`  | `5151`  | Port to launch the FiftyOne app on.      |
| `skip_launch` | `bool` | `false` | Skip launching FiftyOne after processing.|
| `verbose`     | `bool` | `false` | Enable debug logging.                    |

### Config file

| Option   | Type  | Default  | Description                                                        |
| -------- | ----- | -------- | ------------------------------------------------------------------ |
| `config` | `str` | *(none)* | Path to config YAML file. Overrides default settings.              |

## Using a config file

Instead of passing all options on the command line, you can create a YAML config
file with any subset of options (all keys are optional and override the defaults):

```yaml
# my_config.yaml
data: "/path/to/your/dataset"   # directory, data.yaml, or ul://username/datasets/slug
task: "detect"                  # detect, segment, classify, pose, obb
name: "my_dataset"              # auto-generated from path if not set
reload: false
dataset_dir: "yolo_scout/datasets"

skip_embeddings: false
model: "openai_clip"
batch: 16
mask_background: true

thumbnail_dir: "yolo_scout/thumbnails"
thumbnail_width: 800

skip_quality: false

port: 5151
skip_launch: false
verbose: false
```

```bash
yolo-scout config=my_config.yaml
```

!!! tip "CLI overrides"

    CLI arguments take precedence over the config file. You can use a config file for defaults and override specific values:

    ```bash
    yolo-scout config=my_config.yaml model=siglip_base_224 batch=64
    ```

## Data sources

| Format          | Example                              | Notes                                                                              |
| --------------- | ------------------------------------ | ---------------------------------------------------------------------------------- |
| Local directory | `data=/path/to/dataset`              | Standard YOLO directory structure                                                  |
| YAML file       | `data=/path/to/data.yaml`            | Resolves to the parent directory automatically                                     |
| NDJSON file     | `data=/path/to/file.ndjson`          | Pre-downloaded Ultralytics Platform export; images are downloaded and converted to YOLO layout |
| URL             | `data=ul://user/datasets/slug`       | See URL schemes below                                                              |

### URL schemes

| Scheme  | Example                               | Notes                                                                                    |
| ------- | ------------------------------------- | ---------------------------------------------------------------------------------------- |
| `ul://` | `ul://<username>/datasets/<slug>`     | [Ultralytics Platform](https://platform.ultralytics.com), requires `ULTRALYTICS_API_KEY` |

## Dataset directory layouts

This tool supports two common YOLO dataset directory structures:

=== "Type-first"

    Images and labels are organized by type first, then by split.

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

=== "Split-first"

    The dataset is organized by split first, then by type (images/labels).

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

## FiftyOne commands

After you have used YoloScout at least once to visualize a dataset, you can use
these commands to interact with FiftyOne datasets and the app directly:

```bash
# List all datasets
fiftyone datasets list

# Delete a specific dataset by name
fiftyone datasets delete <dataset_name>

# Delete all datasets
python -c "import fiftyone as fo; [fo.delete_dataset(name) for name in fo.list_datasets()]"

# Launch the FiftyOne app
fiftyone app launch

# Launch the FiftyOne app with a specific dataset pre-selected
fiftyone app launch <dataset_name>
```
