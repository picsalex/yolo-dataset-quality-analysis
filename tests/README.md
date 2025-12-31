# YOLO Dataset Quality Analysis - Tests

This directory contains the test suite for the YOLO Dataset Quality Analysis tool.

## 🗂️ Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── fixtures/                # Test dataset management
│   ├── dataset_manager.py   # Downloads datasets from GitHub
│   └── __init__.py
├── unit/                    # Unit tests (fast, no external dependencies)
│   ├── test_core/          # Core module tests (21 tests)
│   ├── test_dataset/       # Dataset parsing & conversion (28 tests)
│   └── test_visualization/ # Visualization utilities (10 tests)
└── integration/             # Integration tests (requires datasets)
    ├── test_dataset_fixtures.py      # Dataset fixture validation (10 tests)
    ├── test_loading_pipeline.py      # Full dataset loading (16 tests)
    ├── test_parser_integration.py    # Real annotation parsing (14 tests)
    ├── test_conversion_pipeline.py   # YOLO→FiftyOne conversion (15 tests)
    └── test_embeddings_pipeline.py   # Embeddings computation (20 tests)
```

## 🚀 Running Tests

### Quick Unit Tests (No Dataset Download)
```bash
# Run fast unit tests only (~59 tests, <5 seconds)
pytest tests/unit/ --skip-download -v

# With coverage
pytest tests/unit/ --skip-download --cov=src --cov-report=html
```

### Integration Tests (Downloads Datasets on First Run)
```bash
# Run integration tests (~75 tests, downloads ~30MB on first run)
pytest tests/integration/ -v

# Run all tests (unit + integration)
pytest -v

# Skip slow tests
pytest -m "not slow" -v
```

### Full Test Suite
```bash
# Run everything with coverage (~134 tests total)
pytest --cov=src --cov-report=html --cov-report=term -v
```

## 📦 Test Datasets

Test datasets are automatically downloaded from GitHub releases on first test run.

### Manual Dataset Management
```bash
# Download datasets manually
python tests/fixtures/dataset_manager.py download

# Show dataset info
python tests/fixtures/dataset_manager.py info

# Clear cached datasets
python tests/fixtures/dataset_manager.py clear

# Compute checksum
python tests/fixtures/dataset_manager.py checksum
```

### Dataset Details
- **Source**: https://github.com/picsalex/yolo-dataset-quality-analysis/releases/download/v1.0.0/datasets.zip
- **Size**: ~30MB
- **Cache Location**: `tests/fixtures/.cache/`
- **Extract Location**: `tests/fixtures/datasets/`

Available datasets:
- `detect_dataset` - Object detection (YOLO format)
- `classify_dataset` - Image classification
- `segment_dataset` - Instance segmentation
- `pose_dataset` - Pose estimation
- `obb_dataset` - Oriented bounding boxes

## 🎯 Test Coverage by Module

### Unit Tests (59 tests)

| Module | Tests | What's Tested |
|--------|-------|---------------|
| **core/enums.py** | 11 | Enum validation, model configs |
| **core/constants.py** | 10 | Field mapping, color palettes |
| **dataset/parser.py** | 18 | YOLO format parsing (all tasks) |
| **dataset/converter.py** | 10 | FiftyOne conversion, bbox clamping |
| **visualization/iou.py** | 10 | IoU computation for boxes & polygons |

### Integration Tests (55 tests)

| Test Suite | Tests | What's Tested |
|------------|-------|---------------|
| **test_dataset_fixtures** | 10 | Dataset download & structure validation |
| **test_loading_pipeline** | 16 | Full dataset loading, caching, metadata |
| **test_parser_integration** | 14 | Real YOLO file parsing, validation |
| **test_conversion_pipeline** | 15 | YOLO→FiftyOne conversion, IoU scores |
| **test_embeddings_pipeline** | 20 | Embeddings computation, field mapping |

**Total: 134 comprehensive tests** ✅

## 🎯 Test Markers

Tests are organized with pytest markers:

- `@pytest.mark.unit` - Unit tests (fast, no dependencies)
- `@pytest.mark.integration` - Integration tests (use real datasets)
- `@pytest.mark.requires_dataset` - Requires dataset download
- `@pytest.mark.slow` - Slow-running tests (>5 seconds)

```bash
# Run only unit tests
pytest -m unit -v

# Run only integration tests
pytest -m integration -v

# Skip slow tests
pytest -m "not slow" -v

# Skip dataset-dependent tests
pytest --skip-download -v
```

## 📊 Coverage Goals

| Module | Target | Current Status |
|--------|--------|----------------|
| core/* | 95% | ✅ Comprehensive |
| dataset/parser.py | 90% | ✅ Comprehensive |
| dataset/converter.py | 90% | ✅ Comprehensive |
| dataset/loader.py | 80% | ✅ Integration tests |
| visualization/iou.py | 85% | ✅ Comprehensive |

## 🔧 Adding New Tests

### Unit Test Template
```python
"""Tests for src.module.feature."""

import pytest
from src.module.feature import function_to_test


class TestFeature:
    """Tests for Feature class/function."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        result = function_to_test(input_data)
        assert result == expected_output

    def test_edge_case(self):
        """Test edge case handling."""
        result = function_to_test(edge_case_input)
        assert result is not None
```

### Integration Test Template
```python
"""Integration tests for module."""

import pytest
import fiftyone as fo


@pytest.mark.requires_dataset
@pytest.mark.integration
class TestIntegration:
    """Integration tests."""

    def test_with_real_data(self, detect_dataset, tmp_path):
        """Test with real dataset."""
        dataset_name = "test_integration"

        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        try:
            # Your test code here
            assert True
        finally:
            # Cleanup
            if dataset_name in fo.list_datasets():
                fo.delete_dataset(dataset_name)
```

## 🐛 Debugging Tests

```bash
# Run with verbose output
pytest -vv

# Stop at first failure
pytest -x

# Show local variables on failure
pytest -l

# Run specific test with debugging
pytest tests/unit/test_core/test_enums.py::TestDatasetTask -vv -s

# Run with print statements visible
pytest -s

# Run last failed tests only
pytest --lf
```

## 💡 What's Being Tested

### Unit Tests Focus On:
✅ **Pure logic** - No external dependencies
✅ **Edge cases** - Empty inputs, boundaries, None values
✅ **Error handling** - Invalid inputs, malformed data
✅ **Data validation** - Format checking, range validation
✅ **Transformations** - Parsing, conversions, calculations

### Integration Tests Focus On:
✅ **Real data** - Actual YOLO files from datasets
✅ **FiftyOne integration** - Dataset creation, sample loading
✅ **End-to-end workflows** - Parse → Convert → Validate
✅ **Data consistency** - Image-label pairing, field presence
✅ **Metadata extraction** - Image info, class names, counts

## 🔄 Continuous Integration

Tests run automatically on GitHub Actions:
- **Unit Tests**: On every push (fast, no dataset download)
- **Integration Tests**: On every push (cached datasets)

See `.github/workflows/test.yml` for CI configuration.

## 📝 Test Development Best Practices

1. **Start with unit tests** - Fast feedback, no dependencies
2. **Mock external dependencies** - Use `unittest.mock` for FiftyOne, cv2
3. **Use fixtures** - Share common test data via pytest fixtures
4. **Test edge cases** - Empty inputs, None values, boundary conditions
5. **Keep tests isolated** - Each test should be independent
6. **Use descriptive names** - Test names should describe what they test
7. **Clean up after integration tests** - Always delete FiftyOne datasets
8. **Test with real data** - Integration tests use actual YOLO datasets

## 🎓 Key Testing Patterns Used

### Dict-Like Objects for Mocking
```python
class DictLikeObject(dict):
    """Supports both obj['key'] and obj.attribute syntax."""
    pass

# Use in tests
obj = DictLikeObject()
obj.bounding_box = [0.2, 0.2, 0.4, 0.4]
obj["iou_score"] = 0.5  # Works!
```

### FiftyOne Dataset Cleanup
```python
@pytest.mark.requires_dataset
def test_something(detect_dataset, tmp_path):
    dataset_name = "test_dataset"

    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    try:
        # Test code
        pass
    finally:
        # Always cleanup
        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)
```

### Session-Scoped Dataset Fixtures
```python
@pytest.fixture(scope="session")
def detect_dataset() -> Path:
    """Downloads once per test session, reused across tests."""
    return get_dataset_path("detect")
```

## 🚀 Running Specific Test Categories

```bash
# Only dataset loading tests
pytest tests/integration/test_loading_pipeline.py -v

# Only parser tests
pytest tests/integration/test_parser_integration.py -v

# Only conversion tests
pytest tests/integration/test_conversion_pipeline.py -v

# Only unit tests for core module
pytest tests/unit/test_core/ -v

# Only IoU tests
pytest tests/unit/test_visualization/test_iou.py -v
```

## ✅ Current Test Status

**Total Tests**: 134
**Unit Tests**: 59 (100% passing)
**Integration Tests**: 75 (100% passing)
**Coverage**: ~85% of src/ directory
**Status**: ✅ All tests passing!

The test suite provides comprehensive coverage of:
- ✅ Core functionality (enums, constants)
- ✅ YOLO annotation parsing (all 5 tasks)
- ✅ FiftyOne conversion (all formats)
- ✅ Dataset loading pipeline
- ✅ Metadata extraction
- ✅ IoU computation
- ✅ Real-world data validation
