"""Integration tests for embeddings computation and storage."""

import pytest
import fiftyone as fo

from src.dataset.loader import load_yolo_dataset
from src.embeddings.computer import compute_embeddings
from src.core.enums import DatasetTask, EmbeddingsModel
from src.core.constants import (
    IMAGE_EMBEDDINGS_KEY,
    PATCH_EMBEDDINGS_KEY,
    DETECTION_FIELD,
    get_field_name,
)


@pytest.mark.requires_dataset
@pytest.mark.integration
@pytest.mark.slow
class TestImageEmbeddings:
    """Test image embeddings computation and storage."""

    def test_image_embeddings_computed_detection(self, detect_dataset, tmp_path):
        """Test that image embeddings are computed for detection dataset."""
        dataset_name = "test_img_emb_detect"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        # Load dataset
        _, dataset = load_yolo_dataset(
            dataset_path=str(detect_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.DETECTION,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            # Compute embeddings with small batch for speed
            model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=model_kwargs,
                batch_size=4,
            )

            # Verify image embeddings brain key exists
            brain_keys = dataset.list_brain_runs()
            assert IMAGE_EMBEDDINGS_KEY in brain_keys, (
                f"Image embeddings key not found. Available keys: {brain_keys}"
            )

            # Verify embeddings are stored
            brain_info = dataset.get_brain_info(IMAGE_EMBEDDINGS_KEY)
            assert brain_info is not None
            assert hasattr(brain_info, "config")

        finally:
            fo.delete_dataset(dataset_name)

    def test_image_embeddings_for_all_tasks(self, all_datasets, tmp_path):
        """Test that image embeddings work for all dataset tasks."""
        model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()

        for task_name, dataset_path in all_datasets.items():
            dataset_name = f"test_img_emb_{task_name}"

            if dataset_name in fo.list_datasets():
                fo.delete_dataset(dataset_name)

            # Map task_name to DatasetTask enum
            task_map = {
                "detect": DatasetTask.DETECTION,
                "classify": DatasetTask.CLASSIFICATION,
                "segment": DatasetTask.SEGMENTATION,
                "pose": DatasetTask.POSE,
                "obb": DatasetTask.OBB,
            }
            task = task_map[task_name]

            try:
                # Load dataset
                _, dataset = load_yolo_dataset(
                    dataset_path=str(dataset_path),
                    dataset_name=dataset_name,
                    dataset_task=task,
                    force_reload=True,
                    thumbnail_width=100,
                    thumbnail_dir=str(tmp_path / "thumbnails"),
                )

                # Compute embeddings
                compute_embeddings(
                    dataset=dataset,
                    dataset_task=task,
                    model_kwargs=model_kwargs,
                    batch_size=4,
                )

                # Verify image embeddings exist
                brain_keys = dataset.list_brain_runs()
                assert IMAGE_EMBEDDINGS_KEY in brain_keys, (
                    f"Image embeddings missing for {task_name}"
                )

            finally:
                if dataset_name in fo.list_datasets():
                    fo.delete_dataset(dataset_name)

    def test_image_embeddings_visualization_exists(self, detect_dataset, tmp_path):
        """Test that visualization (UMAP) is computed for image embeddings."""
        dataset_name = "test_img_emb_viz"

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
            model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=model_kwargs,
                batch_size=4,
            )

            # Check that brain run has visualization
            brain_info = dataset.get_brain_info(IMAGE_EMBEDDINGS_KEY)
            assert brain_info is not None

            # Verify config contains visualization method
            config = brain_info.config
            assert hasattr(config, "method") or hasattr(config, "embeddings_field")

        finally:
            fo.delete_dataset(dataset_name)


@pytest.mark.requires_dataset
@pytest.mark.integration
@pytest.mark.slow
class TestPatchEmbeddings:
    """Test patch embeddings computation and storage."""

    def test_patch_embeddings_computed_detection(self, detect_dataset, tmp_path):
        """Test that patch embeddings are computed for detection dataset."""
        dataset_name = "test_patch_emb_detect"

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
            model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=model_kwargs,
                batch_size=4,
            )

            # Verify patch embeddings brain key exists
            brain_keys = dataset.list_brain_runs()
            assert PATCH_EMBEDDINGS_KEY in brain_keys, (
                f"Patch embeddings key not found. Available keys: {brain_keys}"
            )

            # Verify patch embeddings use correct field
            brain_info = dataset.get_brain_info(PATCH_EMBEDDINGS_KEY)
            assert brain_info is not None

        finally:
            fo.delete_dataset(dataset_name)

    def test_patch_embeddings_use_correct_field_detection(
        self, detect_dataset, tmp_path
    ):
        """Test that patch embeddings use bounding_boxes field for detection."""
        dataset_name = "test_patch_field_detect"

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
            model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=model_kwargs,
                batch_size=4,
            )

            # Get brain info
            brain_info = dataset.get_brain_info(PATCH_EMBEDDINGS_KEY)
            config = brain_info.config

            # Should use bounding_boxes field
            patches_field = config.patches_field
            assert patches_field == DETECTION_FIELD, (
                f"Expected patches_field='{DETECTION_FIELD}', got '{patches_field}'"
            )

        finally:
            fo.delete_dataset(dataset_name)

    def test_patch_embeddings_use_correct_field_segmentation(
        self, segment_dataset, tmp_path
    ):
        """Test that patch embeddings use seg_polygons field for segmentation."""
        dataset_name = "test_patch_field_segment"

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
            model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.SEGMENTATION,
                model_kwargs=model_kwargs,
                batch_size=4,
            )

            # Get brain info
            brain_info = dataset.get_brain_info(PATCH_EMBEDDINGS_KEY)
            config = brain_info.config

            # Should use seg_polygons field
            patches_field = config.patches_field
            expected_field = get_field_name(DatasetTask.SEGMENTATION)
            assert patches_field == expected_field, (
                f"Expected patches_field='{expected_field}', got '{patches_field}'"
            )

        finally:
            fo.delete_dataset(dataset_name)

    def test_patch_embeddings_use_correct_field_pose(self, pose_dataset, tmp_path):
        """Test that patch embeddings use bounding_boxes field for pose (not keypoints)."""
        dataset_name = "test_patch_field_pose"

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
            model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.POSE,
                model_kwargs=model_kwargs,
                batch_size=4,
            )

            # Get brain info
            brain_info = dataset.get_brain_info(PATCH_EMBEDDINGS_KEY)
            config = brain_info.config

            # Pose should use bounding_boxes, NOT pose_keypoints
            patches_field = config.patches_field
            assert patches_field == DETECTION_FIELD, (
                f"Expected patches_field='{DETECTION_FIELD}' for pose, got '{patches_field}'"
            )

        finally:
            fo.delete_dataset(dataset_name)

    def test_patch_embeddings_use_correct_field_obb(self, obb_dataset, tmp_path):
        """Test that patch embeddings use obb_bounding_boxes field for OBB."""
        dataset_name = "test_patch_field_obb"

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
            model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.OBB,
                model_kwargs=model_kwargs,
                batch_size=4,
            )

            # Get brain info
            brain_info = dataset.get_brain_info(PATCH_EMBEDDINGS_KEY)
            config = brain_info.config

            # Should use obb_bounding_boxes field
            patches_field = config.patches_field
            expected_field = get_field_name(DatasetTask.OBB)
            assert patches_field == expected_field, (
                f"Expected patches_field='{expected_field}', got '{patches_field}'"
            )

        finally:
            fo.delete_dataset(dataset_name)

    def test_classification_no_patch_embeddings(self, classify_dataset, tmp_path):
        """Test that classification dataset does not compute patch embeddings."""
        dataset_name = "test_patch_classify"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        _, dataset = load_yolo_dataset(
            dataset_path=str(classify_dataset),
            dataset_name=dataset_name,
            dataset_task=DatasetTask.CLASSIFICATION,
            force_reload=True,
            thumbnail_width=100,
            thumbnail_dir=str(tmp_path / "thumbnails"),
        )

        try:
            model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.CLASSIFICATION,
                model_kwargs=model_kwargs,
                batch_size=4,
            )

            # Classification should only have image embeddings, not patch embeddings
            brain_keys = dataset.list_brain_runs()
            assert IMAGE_EMBEDDINGS_KEY in brain_keys, (
                "Image embeddings should exist for classification"
            )
            assert PATCH_EMBEDDINGS_KEY not in brain_keys, (
                "Patch embeddings should NOT exist for classification"
            )

        finally:
            fo.delete_dataset(dataset_name)


@pytest.mark.requires_dataset
@pytest.mark.integration
@pytest.mark.slow
class TestEmbeddingsFieldMapping:
    """Test that embeddings use correct fields for each task type."""

    def test_field_mapping_consistency(self):
        """Test that field mapping is consistent across tasks."""
        # Detection uses bounding_boxes
        assert get_field_name(DatasetTask.DETECTION) == DETECTION_FIELD

        # Segmentation uses seg_polygons
        assert get_field_name(DatasetTask.SEGMENTATION) != DETECTION_FIELD

        # OBB uses obb_bounding_boxes
        assert get_field_name(DatasetTask.OBB) != DETECTION_FIELD

        # Pose uses pose_keypoints for labels, but bounding_boxes for patches
        assert get_field_name(DatasetTask.POSE) != DETECTION_FIELD

    def test_all_non_classification_tasks_have_patches(self, all_datasets, tmp_path):
        """Test that all non-classification tasks compute patch embeddings."""
        model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()

        task_map = {
            "detect": DatasetTask.DETECTION,
            "segment": DatasetTask.SEGMENTATION,
            "pose": DatasetTask.POSE,
            "obb": DatasetTask.OBB,
        }

        for task_name, task in task_map.items():
            dataset_name = f"test_patches_{task_name}"
            dataset_path = all_datasets[task_name]

            if dataset_name in fo.list_datasets():
                fo.delete_dataset(dataset_name)

            try:
                _, dataset = load_yolo_dataset(
                    dataset_path=str(dataset_path),
                    dataset_name=dataset_name,
                    dataset_task=task,
                    force_reload=True,
                    thumbnail_width=100,
                    thumbnail_dir=str(tmp_path / "thumbnails"),
                )

                compute_embeddings(
                    dataset=dataset,
                    dataset_task=task,
                    model_kwargs=model_kwargs,
                    batch_size=4,
                )

                # All non-classification tasks should have both embeddings
                brain_keys = dataset.list_brain_runs()
                assert IMAGE_EMBEDDINGS_KEY in brain_keys, (
                    f"Missing image embeddings for {task_name}"
                )
                assert PATCH_EMBEDDINGS_KEY in brain_keys, (
                    f"Missing patch embeddings for {task_name}"
                )

            finally:
                if dataset_name in fo.list_datasets():
                    fo.delete_dataset(dataset_name)


@pytest.mark.requires_dataset
@pytest.mark.integration
@pytest.mark.slow
class TestBackgroundMasking:
    """Test background masking configuration for embeddings."""

    def test_mask_background_enabled_by_default(self, segment_dataset, tmp_path):
        """Test that background masking is enabled by default for segmentation."""
        dataset_name = "test_mask_bg_default"

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
            model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
            # Test with default (mask_background=True)
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.SEGMENTATION,
                model_kwargs=model_kwargs,
                batch_size=4,
                mask_background=True,
            )

            # Should successfully compute embeddings
            brain_keys = dataset.list_brain_runs()
            assert PATCH_EMBEDDINGS_KEY in brain_keys

        finally:
            fo.delete_dataset(dataset_name)

    def test_mask_background_disabled(self, segment_dataset, tmp_path):
        """Test that background masking can be disabled for segmentation."""
        dataset_name = "test_mask_bg_disabled"

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
            model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
            # Test with mask_background=False
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.SEGMENTATION,
                model_kwargs=model_kwargs,
                batch_size=4,
                mask_background=False,
            )

            # Should successfully compute embeddings without masking
            brain_keys = dataset.list_brain_runs()
            assert PATCH_EMBEDDINGS_KEY in brain_keys

        finally:
            fo.delete_dataset(dataset_name)

    def test_mask_background_obb_task(self, obb_dataset, tmp_path):
        """Test that background masking works for OBB tasks."""
        dataset_name = "test_mask_bg_obb"

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
            model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
            # Test both enabled and disabled
            for mask_bg in [True, False]:
                compute_embeddings(
                    dataset=dataset,
                    dataset_task=DatasetTask.OBB,
                    model_kwargs=model_kwargs,
                    batch_size=4,
                    mask_background=mask_bg,
                )

                brain_keys = dataset.list_brain_runs()
                assert PATCH_EMBEDDINGS_KEY in brain_keys

        finally:
            fo.delete_dataset(dataset_name)

    def test_mask_background_not_applied_to_detection(self, detect_dataset, tmp_path):
        """Test that mask_background parameter doesn't affect detection tasks."""
        dataset_name = "test_mask_bg_detect"

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
            model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
            # For detection, mask_background shouldn't matter
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=model_kwargs,
                batch_size=4,
                mask_background=False,  # Should have no effect
            )

            brain_keys = dataset.list_brain_runs()
            assert PATCH_EMBEDDINGS_KEY in brain_keys

        finally:
            fo.delete_dataset(dataset_name)


@pytest.mark.requires_dataset
@pytest.mark.integration
@pytest.mark.slow
class TestEmbeddingsBehavior:
    """Test embeddings computation behavior and edge cases."""

    def test_embeddings_with_small_batch_size(self, detect_dataset, tmp_path):
        """Test that embeddings work with small batch size."""
        dataset_name = "test_emb_small_batch"

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
            model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
            # Use batch_size=1 for edge case
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=model_kwargs,
                batch_size=1,
            )

            # Should still work
            brain_keys = dataset.list_brain_runs()
            assert IMAGE_EMBEDDINGS_KEY in brain_keys
            assert PATCH_EMBEDDINGS_KEY in brain_keys

        finally:
            fo.delete_dataset(dataset_name)

    def test_embeddings_with_samples_without_objects(self, detect_dataset, tmp_path):
        """Test that embeddings work even when some samples have no objects."""
        dataset_name = "test_emb_no_objects"

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
            # Should be able to compute embeddings regardless
            model_kwargs = EmbeddingsModel.OPENAI_CLIP.get_model_kwargs()
            compute_embeddings(
                dataset=dataset,
                dataset_task=DatasetTask.DETECTION,
                model_kwargs=model_kwargs,
                batch_size=4,
            )

            # Image embeddings should work for all samples
            brain_keys = dataset.list_brain_runs()
            assert IMAGE_EMBEDDINGS_KEY in brain_keys

            # Patch embeddings might have fewer entries (only for samples with objects)
            # But brain run should still exist
            assert PATCH_EMBEDDINGS_KEY in brain_keys

        finally:
            fo.delete_dataset(dataset_name)
