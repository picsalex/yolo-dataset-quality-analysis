"""
Logging configuration for YOLO Dataset Quality Analysis Tool
"""

import logging
import sys
import warnings


def setup_logger(
    name: str = "yolo_analysis", level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting

    Args:
        name: Logger name
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent propagation to root logger to avoid duplication
    logger.propagate = False

    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Custom formatter that only shows level prefix for warnings and errors
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            # Only modify messages from our logger, not external ones
            if record.name == name:
                if record.levelno == logging.ERROR:
                    record.msg = f"ERROR ❌ {record.msg}"
                elif record.levelno == logging.WARNING:
                    record.msg = f"WARNING ⚠️  {record.msg}"
            return super().format(record)

    formatter = CustomFormatter("%(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


def configure_external_loggers(level: int = logging.WARNING) -> None:
    """
    Configure logging levels for external libraries to reduce noise

    Args:
        level: Logging level for external libraries (default: WARNING)
    """
    # Suppress verbose output from external libraries
    logging.getLogger("fiftyone").setLevel(level)
    logging.getLogger("PIL").setLevel(level)
    logging.getLogger("matplotlib").setLevel(level)
    logging.getLogger("urllib3").setLevel(level)
    logging.getLogger("requests").setLevel(level)
    logging.getLogger("eta").setLevel(level)
    logging.getLogger("eta.core.utils").setLevel(level)

    # Prevent propagation from external loggers
    for logger_name in [
        "fiftyone",
        "PIL",
        "matplotlib",
        "urllib3",
        "requests",
        "eta",
        "eta.core.utils",
    ]:
        ext_logger = logging.getLogger(logger_name)
        ext_logger.propagate = False

    # Suppress the QuickGELU mismatch warning from open_clip
    warnings.filterwarnings(
        "ignore",
        message="QuickGELU mismatch.*",
        category=UserWarning,
        module="open_clip.factory",
    )


# Global logger instance
logger = setup_logger()
