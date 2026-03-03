"""Compute image quality metrics for FiftyOne datasets."""

from functools import partial
from multiprocessing import Pool, cpu_count

import cv2
import fiftyone as fo
import numpy as np
from tqdm import tqdm

from src.core.constants import DETECTION_FIELD, get_field_name
from src.core.enums import DatasetTask
from src.embeddings.preprocessing import process_sample_patches
from src.utils.logger import logger


def _blurriness(gray: np.ndarray) -> float:
    """
    Laplacian variance — lower = blurrier.
    Returns the inverse so higher = blurrier for easier interpretation.
    """
    return 1.0 / (1.0 + cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_quality_metrics(
    dataset: fo.Dataset,
    dataset_task: DatasetTask,
    mask_background: bool,
) -> None:
    """Compute quality metrics for images and patches."""
    logger.info("Computing quality metrics...")

    import time

    start = time.time()

    # Image-level
    for sample in tqdm(dataset, desc="Image metrics"):
        gray = cv2.imread(sample.filepath, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        sample["blurriness"] = _blurriness(gray)
        sample.save()

    # Patch-level
    if dataset_task == DatasetTask.CLASSIFICATION:
        return

    patches_field = get_field_name(task=dataset_task)
    if dataset_task == DatasetTask.POSE:
        patches_field = DETECTION_FIELD

    is_detection_like = dataset_task in [DatasetTask.DETECTION, DatasetTask.POSE]

    def get_patches(sample):
        obj = sample[patches_field]
        if obj is None:
            return []
        return (obj.detections if is_detection_like else obj.polylines) or []

    sample_data_list = [
        (s.id, s.filepath, patches_field, get_patches(s), dataset_task)
        for s in dataset.select_fields([patches_field, "filepath"])
        if get_patches(s)
    ]

    if not sample_data_list:
        return

    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = list(
            tqdm(
                pool.imap(
                    partial(process_sample_patches, mask_background=mask_background),
                    sample_data_list,
                ),
                total=len(sample_data_list),
                desc="Patch metrics",
            )
        )

    for (sample_id, *_), (_, crops) in zip(sample_data_list, results):
        sample = dataset[sample_id]
        patches = get_patches(sample)
        for patch, crop in zip(patches, crops):
            patch["blurriness"] = _blurriness(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY))
        sample.save()

    logger.info("Quality metrics computed successfully")
    end = time.time()
    print(f"Quality metrics computation took {end - start:.2f} seconds")
