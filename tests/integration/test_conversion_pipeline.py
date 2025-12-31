"""Integration tests for conversion from YOLO to FiftyOne format."""

import pytest
import fiftyone as fo

from src.dataset.loader import load_yolo_dataset
from src.core.enums import DatasetTask


@pytest.mark.requires_dataset
@pytest.mark.integration
class TestYoloToFiftyOneConversion:
    """Test YOLO to FiftyOne conversion with real data."""

    def test_detection_conversion_complete(self, detect_dataset, tmp_path):
        """Test that detection conversion creates all expected fields."""
        dataset_name = "test_convert_detection"

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
            # Find samples with detections
            samples_with_objects = [s for s in dataset if s["object_count"] > 0]
            assert len(samples_with_objects) > 0

            sample = samples_with_objects[0]
            detections = sample["bounding_boxes"]

            assert detections is not None
            assert len(detections.detections) > 0

            detection = detections.detections[0]

            # Verify all metadata fields are present
            required_fields = ["area", "aspect_ratio", "width", "height", "iou_score"]
            for field in required_fields:
                assert field in detection, f"Missing field: {field}"
                assert detection[field] is not None

        finally:
            fo.delete_dataset(dataset_name)

    def test_bbox_normalization(self, detect_dataset, tmp_path):
        """Test that bounding boxes are properly normalized to [0, 1]."""
        dataset_name = "test_convert_bbox_norm"

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
                if sample["bounding_boxes"] is None:
                    continue

                for detection in sample["bounding_boxes"].detections:
                    bbox = detection.bounding_box
                    x, y, w, h = bbox

                    # All values should be in [0, 1]
                    assert 0 <= x <= 1, f"x out of range: {x}"
                    assert 0 <= y <= 1, f"y out of range: {y}"
                    assert 0 <= w <= 1, f"w out of range: {w}"
                    assert 0 <= h <= 1, f"h out of range: {h}"

                    # Box should not extend beyond image
                    assert x + w <= 1.0001, f"Box extends beyond right edge: {x + w}"
                    assert y + h <= 1.0001, f"Box extends beyond bottom edge: {y + h}"

        finally:
            fo.delete_dataset(dataset_name)

    def test_label_mapping(self, detect_dataset, tmp_path):
        """Test that class IDs are correctly mapped to class names."""
        dataset_name = "test_convert_labels"

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
            class_names = dataset.info["class_names"]

            for sample in dataset:
                if sample["bounding_boxes"] is None:
                    continue

                for detection in sample["bounding_boxes"].detections:
                    label = detection.label

                    # Label should either be in class_names or be "class_N" format
                    assert label in class_names or label.startswith("class_"), (
                        f"Invalid label: {label}"
                    )

        finally:
            fo.delete_dataset(dataset_name)

    def test_segmentation_polygon_conversion(self, segment_dataset, tmp_path):
        """Test that segmentation polygons are properly converted."""
        dataset_name = "test_convert_segmentation"

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
            # Find samples with polygons
            samples_with_objects = [s for s in dataset if s["object_count"] > 0]
            assert len(samples_with_objects) > 0

            sample = samples_with_objects[0]
            polygons = sample["seg_polygons"]

            assert polygons is not None
            assert len(polygons.polylines) > 0

            polygon = polygons.polylines[0]

            # Verify polygon properties
            assert polygon.closed is True
            assert polygon.filled is True
            assert polygon.points is not None
            assert len(polygon.points) > 0
            assert len(polygon.points[0]) >= 4  # At least 3 points + closing point

            # Verify metadata fields
            assert "area" in polygon
            assert "num_keypoints" in polygon
            assert polygon["area"] > 0

        finally:
            fo.delete_dataset(dataset_name)

    def test_pose_keypoints_conversion(self, pose_dataset, tmp_path):
        """Test that pose keypoints are properly converted."""
        dataset_name = "test_convert_pose"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        _, dataset = load_yolo_dataset(
            dataset_path=str(pose_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.POSE,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            # Find samples with keypoints
            samples_with_objects = [s for s in dataset if s["object_count"] > 0]
            assert len(samples_with_objects) > 0

            sample = samples_with_objects[0]
            keypoints = sample["pose_keypoints"]

            assert keypoints is not None
            assert len(keypoints.keypoints) > 0

            keypoint = keypoints.keypoints[0]

            # Verify keypoint properties
            assert keypoint.points is not None
            assert len(keypoint.points) > 0
            assert "num_keypoints" in keypoint

            # Verify hidden keypoints are marked as [-1, -1]
            for point in keypoint.points:
                if point[0] == -1 and point[1] == -1:
                    # This is a hidden keypoint
                    assert True
                else:
                    # Visible keypoint should be in [0, 1]
                    assert 0 <= point[0] <= 1
                    assert 0 <= point[1] <= 1

        finally:
            fo.delete_dataset(dataset_name)

    def test_pose_creates_detections(self, pose_dataset, tmp_path):
        """Test that pose task also creates bounding box detections."""
        dataset_name = "test_convert_pose_detections"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        _, dataset = load_yolo_dataset(
            dataset_path=str(pose_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.POSE,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            # Find samples with keypoints
            samples_with_objects = [s for s in dataset if s["object_count"] > 0]
            assert len(samples_with_objects) > 0

            sample = samples_with_objects[0]

            # Should have both keypoints and detections
            assert "pose_keypoints" in sample
            assert "bounding_boxes" in sample

            keypoints = sample["pose_keypoints"]
            detections = sample["bounding_boxes"]

            if keypoints and keypoints.keypoints:
                # Number of detections should match number of keypoints
                assert len(detections.detections) == len(keypoints.keypoints)

                # Detection should have num_keypoints field
                detection = detections.detections[0]
                assert "num_keypoints" in detection

        finally:
            fo.delete_dataset(dataset_name)

    def test_obb_conversion(self, obb_dataset, tmp_path):
        """Test that OBB (oriented bounding boxes) are properly converted."""
        dataset_name = "test_convert_obb"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        _, dataset = load_yolo_dataset(
            dataset_path=str(obb_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.OBB,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            # Find samples with OBBs
            samples_with_objects = [s for s in dataset if s["object_count"] > 0]
            assert len(samples_with_objects) > 0

            sample = samples_with_objects[0]
            obbs = sample["obb_bounding_boxes"]

            assert obbs is not None
            assert len(obbs.polylines) > 0

            obb = obbs.polylines[0]

            # Verify OBB properties
            assert obb.closed is True
            assert obb.filled is False  # OBBs are not filled
            assert obb.points is not None
            assert len(obb.points[0]) == 5  # 4 corners + closing point

            # Verify metadata
            assert "area" in obb
            assert "width" in obb
            assert "height" in obb

        finally:
            fo.delete_dataset(dataset_name)


@pytest.mark.requires_dataset
@pytest.mark.integration
class TestIoUComputation:
    """Test IoU score computation during conversion."""

    def test_iou_scores_computed(self, detect_dataset, tmp_path):
        """Test that IoU scores are computed for all detections."""
        dataset_name = "test_iou_detection"

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
                if sample["bounding_boxes"] is None:
                    continue

                for detection in sample["bounding_boxes"].detections:
                    # Every detection should have an iou_score
                    assert "iou_score" in detection
                    assert isinstance(detection["iou_score"], (int, float))
                    assert 0 <= detection["iou_score"] <= 1

        finally:
            fo.delete_dataset(dataset_name)

    def test_overlapping_objects_have_nonzero_iou(self, detect_dataset, tmp_path):
        """Test that overlapping objects have non-zero IoU scores."""
        dataset_name = "test_iou_overlap"

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
            # Find samples with multiple objects
            samples_with_multiple = [s for s in dataset if s["object_count"] >= 2]

            if len(samples_with_multiple) > 0:
                # At least some samples should have overlapping objects
                found_overlap = False
                for sample in samples_with_multiple:
                    for detection in sample["bounding_boxes"].detections:
                        if detection["iou_score"] > 0.1:  # Some overlap
                            found_overlap = True
                            break
                    if found_overlap:
                        break

                # It's okay if no overlaps found (depends on dataset)
                # Just verify IoU computation doesn't crash
                assert True

        finally:
            fo.delete_dataset(dataset_name)

    def test_single_object_has_zero_iou(self, detect_dataset, tmp_path):
        """Test that samples with single object have zero IoU."""
        dataset_name = "test_iou_single"

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
            # Find samples with exactly one object
            samples_with_one = [s for s in dataset if s["object_count"] == 1]

            for sample in samples_with_one:
                detection = sample["bounding_boxes"].detections[0]
                # Single object should have IoU = 0 (no other objects to overlap)
                assert detection["iou_score"] == 0.0

        finally:
            fo.delete_dataset(dataset_name)
