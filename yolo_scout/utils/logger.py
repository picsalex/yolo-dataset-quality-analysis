"""Logging configuration for YOLO Dataset Quality Analysis Tool."""

import logging


def setup_logger(name: str = "yolo_analysis", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.

    Args:
        name: Logger name
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    import sys

    if any(arg in ("verbose", "verbose=true", "verbose=True", "verbose=1") for arg in sys.argv[1:]):
        level = logging.DEBUG
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
                    record.msg = f"ERROR {record.msg}"
                elif record.levelno == logging.WARNING:
                    record.msg = f"WARNING {record.msg}"
            return super().format(record)

    formatter = CustomFormatter("%(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


def configure_external_loggers(level: int = logging.WARNING) -> None:
    """Configure logging levels for external libraries to reduce noise."""
    for name in ["fiftyone", "PIL", "matplotlib", "urllib3", "requests", "eta", "eta.core.utils"]:
        ext_logger = logging.getLogger(name)
        ext_logger.setLevel(level)
        ext_logger.propagate = False


# Global logger instance
logger = setup_logger()
