"""Integration tests for dataset loading."""

import pytest


@pytest.mark.requires_dataset
@pytest.mark.integration
class TestDatasetFixtures:
    """Test that dataset fixtures work correctly."""

    def test_detect_dataset_exists(self, detect_dataset):
        """Test that detection dataset is downloaded and accessible."""
        assert detect_dataset.exists()
        assert detect_dataset.is_dir()
        assert (detect_dataset / "data.yaml").exists()

    def test_classify_dataset_exists(self, classify_dataset):
        """Test that classification dataset is downloaded and accessible."""
        assert classify_dataset.exists()
        assert classify_dataset.is_dir()

    def test_segment_dataset_exists(self, segment_dataset):
        """Test that segmentation dataset is downloaded and accessible."""
        assert segment_dataset.exists()
        assert segment_dataset.is_dir()
        assert (segment_dataset / "data.yaml").exists()

    def test_pose_dataset_exists(self, pose_dataset):
        """Test that pose dataset is downloaded and accessible."""
        assert pose_dataset.exists()
        assert pose_dataset.is_dir()
        assert (pose_dataset / "data.yaml").exists()

    def test_obb_dataset_exists(self, obb_dataset):
        """Test that OBB dataset is downloaded and accessible."""
        assert obb_dataset.exists()
        assert obb_dataset.is_dir()
        assert (obb_dataset / "data.yaml").exists()

    def test_all_datasets_fixture(self, all_datasets):
        """Test that all_datasets fixture provides all dataset types."""
        expected_types = ["detect", "classify", "segment", "pose", "obb"]

        for dataset_type in expected_types:
            assert dataset_type in all_datasets
            assert all_datasets[dataset_type].exists()

    def test_detect_dataset_structure(self, detect_dataset):
        """Test detection dataset has expected structure."""
        # Check for image and label directories
        images_train = detect_dataset / "images" / "train"
        labels_train = detect_dataset / "labels" / "train"

        assert images_train.exists(), "Missing train images directory"
        assert labels_train.exists(), "Missing train labels directory"

        # Check for at least one image
        train_images = list(images_train.glob("*.jpg")) + list(
            images_train.glob("*.png")
        )
        assert len(train_images) > 0, "No training images found"

    def test_classify_dataset_structure(self, classify_dataset):
        """Test classification dataset has expected structure."""
        # Should have train/val directories
        train_dir = classify_dataset / "train"
        assert train_dir.exists(), "Missing train directory"

        # Should have class subdirectories
        class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        assert len(class_dirs) > 0, "No class directories found"

        # Check for images in class directories
        for class_dir in class_dirs:
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            assert len(images) > 0, f"No images in {class_dir.name}"


@pytest.mark.integration
class TestDatasetManager:
    """Test dataset manager utilities."""

    def test_get_dataset_path(self):
        """Test get_dataset_path helper function."""
        from tests.fixtures.dataset_manager import get_dataset_path

        detect_path = get_dataset_path("detect")
        assert detect_path.exists()
        assert detect_path.name == "detect_dataset"

    def test_get_dataset_path_invalid_type(self):
        """Test that invalid dataset type raises error."""
        from tests.fixtures.dataset_manager import get_dataset_path

        with pytest.raises(ValueError, match="Unknown dataset type"):
            get_dataset_path("invalid_type")

    def test_datasets_root(self, datasets_root):
        """Test that datasets_root fixture provides root directory."""
        assert datasets_root.exists()
        assert datasets_root.is_dir()

        # Should contain all dataset directories
        dataset_dirs = [d.name for d in datasets_root.iterdir() if d.is_dir()]
        assert "detect_dataset" in dataset_dirs or len(dataset_dirs) > 0
