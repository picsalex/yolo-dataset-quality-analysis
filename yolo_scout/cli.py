#!/usr/bin/env python3
"""
YoloScout — YOLO Dataset Quality Analysis Tool
CLI entry point

Author: Alexis Béduneau
License: MIT
"""

from yolo_scout.utils.decorators import pipeline
from yolo_scout.utils.logger import configure_external_loggers, logger
from yolo_scout.pipeline.steps import (
    validate,
    prepare_run,
    prepare_plugins,
    run_load_dataset,
    run_embeddings,
    run_quality_metrics,
    run_thumbnails,
)


@pipeline
def main():
    configure_external_loggers()

    logger.info("=" * 60)
    logger.info("FIFTYONE YOLO DATASET ANALYSIS")
    logger.info("=" * 60)

    config = validate(verbose=True)
    recompute = prepare_run(config=config)

    prepare_plugins()

    dataset = run_load_dataset(config=config)
    run_embeddings(dataset=dataset, config=config, recompute=recompute)
    run_quality_metrics(dataset=dataset, config=config, recompute=recompute)
    run_thumbnails(dataset=dataset, config=config, recompute=recompute)

    from yolo_scout.visualization.fiftyone_ops import launch_fiftyone_app

    if not config.skip_launch:
        launch_fiftyone_app(dataset=dataset, dataset_task=config.task, port=config.port)
    else:
        logger.info(f"\nProcessing complete. Dataset saved as: {config.name}")
        logger.info("To launch the app later, run:")
        logger.info(f"    fiftyone app launch {config.name}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
