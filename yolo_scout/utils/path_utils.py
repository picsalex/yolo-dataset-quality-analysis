"""Path utility functions."""

from pathlib import PurePosixPath


def get_image_name(path: str) -> str:
    """
    Extract the filename from a path, working for paths from any OS.

    Uses PurePosixPath after normalizing backslashes so that Windows-style
    paths (e.g. C:\\Users\\img.jpg) are handled correctly on any platform.

    Args:
        path: Absolute or relative file path (POSIX or Windows style)

    Returns:
        Filename with extension, e.g. "image.jpg"
    """
    return PurePosixPath(path.replace("\\", "/")).name
