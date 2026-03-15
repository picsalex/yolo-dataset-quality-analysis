import sys


def main():
    """
    Wraps the ultralytics `yolo` CLI and adds the `scout` subcommand.

    - `yolo scout [args]`  → handled by yolo-scout
    - `yolo <anything>`    → delegated to ultralytics as-is
    """
    args = sys.argv[1:]

    if args and args[0] == "scout":
        sys.argv = ["yolo scout"] + args[1:]
        from yolo_scout.cli import main as scout_main

        scout_main()
    else:
        try:
            from ultralytics.cfg import entrypoint

            entrypoint()
        except ImportError:
            print(
                "ultralytics is not installed. "
                "Install it with: pip install ultralytics\n"
                "Or run yolo-scout directly for dataset analysis."
            )
            sys.exit(1)
