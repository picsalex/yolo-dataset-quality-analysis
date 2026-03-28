---
tags:
  - Reference
  - Setup
---

# Configuration

## Configuration hierarchy

YoloScout uses a three-level configuration system. Each level overrides the
previous:

1. **Defaults** — built-in values from `yolo_scout/cfg/default.yaml`
2. **Config file** — optional YAML file via `config=path.yaml`
3. **CLI arguments** — highest priority, override everything

!!! example "Override chain"

    ```bash
    # default.yaml sets batch=16
    # config.yaml sets batch=32
    # CLI sets batch=64 → final value is 64
    yolo-scout config=config.yaml batch=64
    ```

## Default configuration

The full default configuration:

```yaml
# Dataset settings
data:                              # (required) path, data.yaml, or ul:// URL
task: detect                       # detect | classify | segment | pose | obb
name:                              # auto-generated from path if empty
reload: false                      # force reload cached dataset
dataset_dir: yolo_scout/datasets   # download directory for URL sources

# Embeddings settings
skip_embeddings: false
model: openai_clip                 # openai_clip | metaclip_400m | metaclip_fullcc | siglip_base_224
batch: 16
mask_background: true              # mask background in segment/OBB patches

# Thumbnail settings
thumbnail_dir: yolo_scout/thumbnails
thumbnail_width: 800

# Quality settings
skip_quality: false

# App settings
port: 5151
skip_launch: false
verbose: false
```

## Config file

Create a YAML file with any subset of options (all keys are optional and
override the defaults):

```yaml
# my-config.yaml
data: "/path/to/your/dataset"   # directory, data.yaml, or ul://username/datasets/slug
task: "detect"                  # detect, segment, classify, pose, obb
name: "my_dataset"              # auto-generated from path if not set
model: "metaclip_400m"
batch: 32
port: 8080
```

```bash
yolo-scout config=my-config.yaml
```

## Auto-generated values

| Value          | Generated from                                  |
| -------------- | ----------------------------------------------- |
| `name`         | Directory name of the dataset path              |
| `thumbnail_dir`| Based on dataset name if not specified           |

## FiftyOne integration

YoloScout creates persistent FiftyOne datasets that survive across runs:

- Datasets are stored in FiftyOne's MongoDB database
- The dataset name (auto-generated or specified via `name=`) acts as the cache key
- Use `reload=true` to delete and recreate the dataset

??? example "Managing cached datasets via CLI"

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

??? example "Managing cached datasets via Python"

    ```python
    import fiftyone as fo

    # List all datasets
    print(fo.list_datasets())

    # Load a specific dataset
    dataset = fo.load_dataset("my-dataset")

    # Delete a specific dataset
    fo.delete_dataset("my-dataset")

    # Delete all datasets
    for name in fo.list_datasets():
        fo.delete_dataset(name)
    ```
