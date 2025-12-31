"""Integration tests for dataset loading pipeline."""

import pytest
import fiftyone as fo

from src.dataset.loader import load_yolo_dataset
from src.core.enums import DatasetTask


@pytest.mark.requires_dataset
@pytest.mark.integration
class TestDatasetLoading:
    """Integration tests for dataset loading."""

    def test_load_detection_dataset(self, detect_dataset, tmp_path):
        """Test loading a real detection dataset."""
        dataset_name = "test_detect_integration"

        # Clean up if exists
        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        was_cached, dataset = load_yolo_dataset(
            dataset_path=str(detect_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.DETECTION,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            # Verify dataset was created
            assert not was_cached
            assert dataset is not None
            assert dataset.name == dataset_name
            assert len(dataset) > 0

            # Verify samples have expected fields
            sample = dataset.first()
            assert sample is not None
            assert "bounding_boxes" in sample
            assert "image_path" in sample
            assert "object_count" in sample

            # Verify class names in metadata
            assert "class_names" in dataset.info
            assert len(dataset.info["class_names"]) > 0

        finally:
            # Cleanup
            fo.delete_dataset(dataset_name)

    def test_load_classification_dataset(self, classify_dataset, tmp_path):
        """Test loading a real classification dataset."""
        dataset_name = "test_classify_integration"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        was_cached, dataset = load_yolo_dataset(
            dataset_path=str(classify_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.CLASSIFICATION,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            assert not was_cached
            assert dataset is not None
            assert len(dataset) > 0

            # Verify samples have classification labels
            sample = dataset.first()
            assert sample is not None
            assert "cls_label" in sample
            assert sample["cls_label"] is not None

        finally:
            fo.delete_dataset(dataset_name)

    def test_load_segmentation_dataset(self, segment_dataset, tmp_path):
        """Test loading a real segmentation dataset."""
        dataset_name = "test_segment_integration"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        was_cached, dataset = load_yolo_dataset(
            dataset_path=str(segment_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.SEGMENTATION,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            assert not was_cached
            assert dataset is not None
            assert len(dataset) > 0

            # Verify samples have segmentation polygons
            sample = dataset.first()
            assert sample is not None
            assert "seg_polygons" in sample

        finally:
            fo.delete_dataset(dataset_name)

    def test_load_pose_dataset(self, pose_dataset, tmp_path):
        """Test loading a real pose dataset."""
        dataset_name = "test_pose_integration"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        was_cached, dataset = load_yolo_dataset(
            dataset_path=str(pose_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.POSE,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            assert not was_cached
            assert dataset is not None
            assert len(dataset) > 0

            # Verify samples have pose keypoints and detections
            sample = dataset.first()
            assert sample is not None
            assert "pose_keypoints" in sample
            assert "bounding_boxes" in sample  # Pose also creates detections

        finally:
            fo.delete_dataset(dataset_name)

    def test_load_obb_dataset(self, obb_dataset, tmp_path):
        """Test loading a real OBB dataset."""
        dataset_name = "test_obb_integration"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        was_cached, dataset = load_yolo_dataset(
            dataset_path=str(obb_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.OBB,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            assert not was_cached
            assert dataset is not None
            assert len(dataset) > 0

            # Verify samples have OBB polygons
            sample = dataset.first()
            assert sample is not None
            assert "obb_bounding_boxes" in sample

        finally:
            fo.delete_dataset(dataset_name)

    def test_dataset_caching(self, detect_dataset, tmp_path):
        """Test that dataset loading is cached correctly."""
        dataset_name = "test_detect_caching"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        # First load - should create new
        was_cached1, dataset1 = load_yolo_dataset(
            dataset_path=str(detect_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.DETECTION,
            force_reload=False,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        # Second load - should use cache
        was_cached2, dataset2 = load_yolo_dataset(
            dataset_path=str(detect_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.DETECTION,
            force_reload=False,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            assert not was_cached1  # First load creates new
            assert was_cached2  # Second load uses cache
            assert dataset1.name == dataset2.name

        finally:
            fo.delete_dataset(dataset_name)

    def test_force_reload(self, detect_dataset, tmp_path):
        """Test that force_reload recreates the dataset."""
        dataset_name = "test_detect_force_reload"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        # First load
        load_yolo_dataset(
            dataset_path=str(detect_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.DETECTION,
            force_reload=False,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        # Force reload
        was_cached, dataset = load_yolo_dataset(
            dataset_path=str(detect_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.DETECTION,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            assert not was_cached  # Force reload should recreate

        finally:
            fo.delete_dataset(dataset_name)


@pytest.mark.requires_dataset
@pytest.mark.integration
class TestDatasetStructure:
    """Test that loaded datasets have correct structure."""

    def test_detection_dataset_fields(self, detect_dataset, tmp_path):
        """Test that detection dataset has all expected fields."""
        dataset_name = "test_detect_fields"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        _, dataset = load_yolo_dataset(
            dataset_path=str(detect_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.DETECTION,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            # Get a sample with detections
            samples_with_detections = [s for s in dataset if s["object_count"] > 0]
            assert len(samples_with_detections) > 0

            sample = samples_with_detections[0]
            detections = sample["bounding_boxes"]

            if detections and len(detections.detections) > 0:
                detection = detections.detections[0]

                # Check detection has expected fields
                assert hasattr(detection, "label")
                assert hasattr(detection, "bounding_box")
                assert "area" in detection
                assert "aspect_ratio" in detection
                assert "width" in detection
                assert "height" in detection
                assert "iou_score" in detection

        finally:
            fo.delete_dataset(dataset_name)

    def test_segmentation_dataset_fields(self, segment_dataset, tmp_path):
        """Test that segmentation dataset has all expected fields."""
        dataset_name = "test_segment_fields"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        _, dataset = load_yolo_dataset(
            dataset_path=str(segment_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.SEGMENTATION,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            # Get a sample with polygons
            samples_with_polygons = [s for s in dataset if s["object_count"] > 0]
            assert len(samples_with_polygons) > 0

            sample = samples_with_polygons[0]
            polygons = sample["seg_polygons"]

            if polygons and len(polygons.polylines) > 0:
                polygon = polygons.polylines[0]

                # Check polygon has expected fields
                assert hasattr(polygon, "label")
                assert hasattr(polygon, "points")
                assert "area" in polygon
                assert "num_keypoints" in polygon
                assert "width" in polygon
                assert "height" in polygon
                assert "iou_score" in polygon

        finally:
            fo.delete_dataset(dataset_name)

    def test_dataset_splits(self, detect_dataset, tmp_path):
        """Test that dataset properly loads different splits."""
        dataset_name = "test_detect_splits"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        _, dataset = load_yolo_dataset(
            dataset_path=str(detect_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.DETECTION,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            # Check that samples have split tags
            all_tags = set()
            for sample in dataset:
                all_tags.update(sample.tags)

            # Should have at least 'train' tag (datasets should have train split)
            assert len(all_tags) > 0
            assert any(tag in all_tags for tag in ["train", "val", "test"])

        finally:
            fo.delete_dataset(dataset_name)


@pytest.mark.requires_dataset
@pytest.mark.integration
class TestDatasetMetadata:
    """Test dataset metadata extraction and storage."""

    def test_class_names_stored(self, detect_dataset, tmp_path):
        """Test that class names are stored in dataset info."""
        dataset_name = "test_detect_metadata"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        _, dataset = load_yolo_dataset(
            dataset_path=str(detect_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.DETECTION,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            # Check class names in metadata
            assert "class_names" in dataset.info
            class_names = dataset.info["class_names"]
            assert isinstance(class_names, list)
            assert len(class_names) > 0

        finally:
            fo.delete_dataset(dataset_name)

    def test_image_metadata_extracted(self, detect_dataset, tmp_path):
        """Test that image metadata is extracted for samples."""
        dataset_name = "test_detect_img_metadata"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        _, dataset = load_yolo_dataset(
            dataset_path=str(detect_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.DETECTION,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            sample = dataset.first()

            # Check metadata exists
            assert sample.metadata is not None
            assert hasattr(sample.metadata, "width")
            assert hasattr(sample.metadata, "height")
            assert sample.metadata.width > 0
            assert sample.metadata.height > 0

        finally:
            fo.delete_dataset(dataset_name)

    def test_object_count_computed(self, detect_dataset, tmp_path):
        """Test that object count is computed for each sample."""
        dataset_name = "test_detect_obj_count"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        _, dataset = load_yolo_dataset(
            dataset_path=str(detect_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.DETECTION,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            for sample in dataset:
                assert "object_count" in sample
                assert isinstance(sample["object_count"], int)
                assert sample["object_count"] >= 0

                # Verify count matches actual detections
                if sample["bounding_boxes"]:
                    actual_count = len(sample["bounding_boxes"].detections)
                    assert sample["object_count"] == actual_count
                else:
                    assert sample["object_count"] == 0

        finally:
            fo.delete_dataset(dataset_name)
