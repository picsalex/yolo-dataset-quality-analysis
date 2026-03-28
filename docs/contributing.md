---
tags:
  - Development
  - Guide
---

# Contributing

Thank you for your interest in contributing to YoloScout!

## Development setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- [pre-commit](https://pre-commit.com/) (for git hooks)

### Getting started

```bash
git clone https://github.com/picsalex/yolo-scout.git
cd yolo-scout
uv sync --group dev
uv run pre-commit install
```

!!! tip

    `pre-commit install` registers the git hooks so they run automatically on
    every `git commit`. You only need to do this once after cloning.

## Development workflow

### Running tests

!!! example "Test commands"

    === "Unit tests (fast)"

        ```bash
        pytest tests/unit/ --skip-download
        ```

    === "Integration tests"

        ```bash
        pytest tests/integration/ -v
        ```

    === "Full suite with coverage"

        ```bash
        pytest --cov=yolo_scout --cov-report=html
        ```

!!! note "Test datasets"

    Integration tests require sample datasets (~30MB) that are downloaded
    automatically on first run and cached in `tests/fixtures/.cache/`.

### Linting and formatting

!!! example "Quality commands"

    === "Lint"

        ```bash
        ruff check .
        ```

    === "Lint + fix"

        ```bash
        ruff check --fix .
        ```

    === "Format"

        ```bash
        ruff format .
        ```

### Pre-commit hooks

Hooks run automatically on every `git commit`. To run them manually:

```bash
uv run pre-commit run --all-files
```

## Project structure

| Directory                   | Purpose                              |
| --------------------------- | ------------------------------------ |
| `yolo_scout/core/`          | Configuration, constants, enums      |
| `yolo_scout/dataset/`       | Dataset loading, parsing, conversion |
| `yolo_scout/embeddings/`    | CLIP model integration               |
| `yolo_scout/visualization/` | Quality metrics, IoU, FiftyOne UI    |
| `yolo_scout/pipeline/`      | Pipeline step orchestration          |
| `yolo_scout/utils/`         | Decorators, logging, plugins         |
| `tests/unit/`               | Unit tests (59 tests, fast)          |
| `tests/integration/`        | Integration tests (75 tests)         |

## Pull requests

1. Fork the repository and create a branch from `main`
2. Make your changes and ensure all tests pass
3. Update documentation if your changes affect public APIs
4. Open a pull request against `main`

!!! success "PR checklist"

    - [x] Tests pass (`pytest tests/unit/ --skip-download`)
    - [x] Linting passes (`ruff check .`)
    - [x] Code is formatted (`ruff format --check .`)
    - [x] Pre-commit hooks pass (`uv run pre-commit run --all-files`)

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](https://github.com/picsalex/yolo-scout/blob/main/LICENSE).
