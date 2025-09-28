# Configuration
import os

from src.config import bounding_boxes_field
from src.dataset import prepare_voxel_dataset
import fiftyone.zoo as foz
import fiftyone as fo

from src.enum import DatasetTask
from src.images import generate_thumbnails
from src.voxel51 import compute_visualizations

# Main configuration
dataset_path = "/Users/alexis/Downloads/foresight/data_mixed_small"  # Change this to your YOLO dataset path
dataset_name = "my_dataset_1"  # Change this to your desired FiftyOne dataset name, will impact thumbnail directory
dataset_task = (
    DatasetTask.DETECTION
)  # Choices are: CLASSIFICATION, DETECTION, SEGMENTATION, POSE and OBB

# Analysis configuration
batch_size = 16
thumbnail_dir = os.path.join(os.getcwd(), "thumbnails", dataset_name)
force_reload = True  # Set to True to force reloading the dataset
clip_model = "clip-vit-base32-torch"  # Available models: https://docs.voxel51.com/model_zoo/models.html


def main():
    print("\n" + "=" * 60)
    print("FIFTYONE YOLO DATASET ANALYSIS")
    print("=" * 60)

    # Step 1: Prepare dataset
    print(f"\n📁 Step 1: Preparing dataset located at: {dataset_path}")
    is_already_loaded, dataset = prepare_voxel_dataset(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        force_reload=force_reload,
        dataset_task=dataset_task,
    )

    if not is_already_loaded:
        # Step 2: Load CLIP model
        print("\n🤖 Step 2: Loading CLIP model...")
        embeddings_model = foz.load_zoo_model(clip_model)
        print(f"✓ Loaded {clip_model}")

        # Step 3: Compute visualizations
        print("\n🧠 Step 3: Computing embeddings and visualizations...")
        compute_visualizations(
            dataset=dataset,
            model=embeddings_model,
            batch_size=batch_size,
            patches_field=bounding_boxes_field,
        )

        # Step 4: Generate thumbnails (after embeddings, before launching app)
        print("\n🖼️ Step 4: Generating thumbnails for optimized UI...")
        generate_thumbnails(
            dataset=dataset,
            thumbnail_dir_path=thumbnail_dir,
        )

    else:
        print("Dataset already loaded, skipping to app launch...")

    # Step 5: Launch app
    print("\n🚀 Step 5: Launching FiftyOne app...")

    session = fo.launch_app(dataset)

    print("Press Ctrl+C to stop the app\n")

    try:
        session.wait()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        print("\n✓ App closed successfully")
        print("=" * 60)


if __name__ == "__main__":
    main()
