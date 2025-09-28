# Configuration
import os

from src.dataset import prepare_voxel_dataset
import fiftyone.zoo as foz
import fiftyone as fo

from src.images import generate_thumbnails
from src.voxel51 import compute_visualizations

DATASET_PATH = "/Users/alexis/Downloads/foresight/data_mixed"  # Change this to your YOLO dataset path
DATASET_NAME = "golf_club"  # Change this to your desired FiftyOne dataset name, will impact thumbnail directory
CLIP_MODEL = "clip-vit-base32-torch"  # Available models: https://docs.voxel51.com/model_zoo/models.html
BATCH_SIZE = 16
THUMBNAIL_DIR = os.path.join(os.getcwd(), "thumbnails", DATASET_NAME)
FORCE_RELOAD = False  # Set to True to force reloading the dataset


def main():
    print("\n" + "=" * 60)
    print("FIFTYONE YOLO DATASET ANALYSIS")
    print("=" * 60)

    # Step 1: Prepare dataset
    print(f"\n📁 Step 1: Preparing dataset located at: {DATASET_PATH}")
    is_already_loaded, dataset = prepare_voxel_dataset(
        dataset_path=DATASET_PATH, dataset_name=DATASET_NAME, FORCE_RELOAD=FORCE_RELOAD
    )

    if not is_already_loaded:
        # Step 2: Load CLIP model
        print("\n🤖 Step 2: Loading CLIP model...")
        embeddings_model = foz.load_zoo_model(CLIP_MODEL)
        print(f"✓ Loaded {CLIP_MODEL}")

        # Step 3: Compute visualizations
        print("\n🧠 Step 3: Computing embeddings and visualizations...")
        compute_visualizations(
            dataset=dataset,
            model=embeddings_model,
            patches_field="ground_truth",
        )

        # Step 4: Generate thumbnails (after embeddings, before launching app)
        print("\n🖼️ Step 4: Generating thumbnails for optimized UI...")
        generate_thumbnails(dataset)

    else:
        print("Dataset already loaded, skipping to app launch...")

    # Step 5: Launch app
    print("\n🚀 Step 5: Launching FiftyOne app...")

    session = fo.launch_app(dataset)

    print("Press Ctrl+C to stop the app\n")

    # Keep app running (like session.wait())
    try:
        session.wait()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        print("\n✓ App closed successfully")
        print("=" * 60)


if __name__ == "__main__":
    main()
