# Testing Guide for FORGE v1

This document describes the comprehensive testing infrastructure for FORGE v1 Neural Audio Workstation, including unit tests, integration tests, and performance benchmarks.

## Table of Contents

- [Quick Start](#quick-start)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Performance Benchmarks](#performance-benchmarks)
- [Writing Tests](#writing-tests)
- [Continuous Integration](#continuous-integration)
- [Coverage Reports](#coverage-reports)

## Quick Start

### Installation

First, install the test dependencies:

```bash
pip install -r requirements-test.txt
```

### Run All Tests (Fast)

Run all tests excluding slow benchmarks:

```bash
./run_tests.sh
# or
python -m pytest tests/ -v -m "not slow"
```

### Run Unit Tests Only

```bash
./run_tests.sh unit
# or
python -m pytest tests/unit/ -v -m "not benchmark"
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                      # Shared fixtures and configuration
├── unit/                            # Unit tests for individual functions
│   ├── test_audio_processing.py    # Audio processing functions
│   ├── test_batch_processing.py    # Batch processing operations
│   ├── test_performance.py         # Performance module tests
│   └── test_utils.py               # Utility functions
└── integration/                     # Integration tests for workflows
    ├── test_workflows.py           # End-to-end workflows
    └── test_speed_benchmarks.py    # Performance benchmarks
```

## Running Tests

### Using the Test Runner Script

The `run_tests.sh` script provides convenient access to different test suites:

```bash
# Show help
./run_tests.sh help

# Run unit tests only
./run_tests.sh unit

# Run integration tests only
./run_tests.sh integration

# Run performance benchmarks
./run_tests.sh benchmark

# Run speed benchmarks
./run_tests.sh speed

# Run fast tests (no slow benchmarks)
./run_tests.sh fast

# Run with coverage report
./run_tests.sh coverage

# Run all tests including slow benchmarks
./run_tests.sh full
```

### Using pytest Directly

You can also use pytest directly with various options:

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/unit/test_utils.py -v

# Run specific test class
python -m pytest tests/unit/test_utils.py::TestUtilityFunctions -v

# Run specific test
python -m pytest tests/unit/test_utils.py::TestUtilityFunctions::test_sanitize_filename -v

# Run tests matching a pattern
python -m pytest tests/ -k "loop" -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run in parallel (faster)
python -m pytest tests/ -n auto
```

## Test Categories

Tests are organized using pytest markers. You can run specific categories:

### Unit Tests

Fast tests for individual functions:

```bash
python -m pytest tests/ -v -m unit
```

### Integration Tests

End-to-end workflow tests:

```bash
python -m pytest tests/ -v -m integration
```

### Benchmark Tests

Performance and speed benchmarks:

```bash
python -m pytest tests/ -v -m benchmark
```

### Slow Tests

Tests that take significant time:

```bash
python -m pytest tests/ -v -m slow
```

### Tests Requiring External Dependencies

```bash
# Tests requiring FFmpeg
python -m pytest tests/ -v -m requires_ffmpeg

# Tests requiring Demucs
python -m pytest tests/ -v -m requires_demucs

# Tests requiring basic-pitch
python -m pytest tests/ -v -m requires_basic_pitch
```

## Performance Benchmarks

### Overview

The performance benchmark suite tests:
- **Speed**: How fast operations complete
- **Memory Usage**: Peak and current memory consumption
- **Scalability**: How performance scales with input size
- **Resource Limits**: CPU and memory usage patterns

### Running Benchmarks

```bash
# Run all benchmarks (may take several minutes)
./run_tests.sh benchmark

# Run specific benchmark categories
python -m pytest tests/integration/test_speed_benchmarks.py -v -m benchmark

# Run unit-level performance tests
python -m pytest tests/unit/test_performance.py -v -m benchmark
```

### Benchmark Categories

#### Loop Extraction Speed
- Tests extraction speed on different audio lengths
- Tests different aperture settings
- Measures memory usage

#### Vocal Chop Generation Speed
- Tests all three modes (silence, onset, hybrid)
- Measures memory usage
- Compares mode performance

#### MIDI Extraction Speed
- Tests MIDI extraction performance
- Requires basic-pitch installation
- Measures memory usage

#### Drum One-Shot Generation Speed
- Tests transient detection speed
- Measures memory usage

#### Batch Processing Speed
- Tests batch loop extraction
- Tests batch chop generation
- Measures per-file average time

#### Utility Performance
- Audio hashing speed
- Audio loading speed
- File operations

#### Memory Usage Tests
- Peak memory tracking
- Memory usage patterns
- Resource limit checks

#### Scalability Tests
- Tests performance scaling with audio length
- Verifies roughly linear scaling
- Detects performance degradation

### Understanding Benchmark Results

Benchmark tests print detailed performance metrics:

```
📊 Loop Extraction (Medium) Performance:
  Time: 3.45s
  Loops: 5
  Memory: 245.32 MB

💾 Batch Processing Memory Usage:
  Peak: 12.3%
  Current: 185.67 MB
  Files processed: 4

📈 Loop Extraction Scaling:
  1.0s audio: 1.23s
  3.0s audio: 2.45s
  5.0s audio: 3.89s
```

### Performance Expectations

The benchmarks include reasonable time limits:
- Loop extraction: < 10s for 5-second audio
- Vocal chop generation: < 5s for 5-second audio
- MIDI extraction: < 30s for 5-second audio
- Drum one-shot generation: < 5s for 5-second audio
- Batch processing: < 20s for 2 files

These limits ensure tests detect performance regressions while accounting for varying system capabilities.

## Writing Tests

### Test Structure

Follow this structure for new tests:

```python
"""
Test module docstring.
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forgev1 import function_to_test


@pytest.mark.unit  # or integration, benchmark, etc.
class TestFeatureName:
    """Test class for feature."""
    
    def test_basic_functionality(self, sample_audio_file):
        """Test basic functionality."""
        result = function_to_test(sample_audio_file)
        assert result is not None
    
    def test_edge_case(self):
        """Test edge case handling."""
        # Test implementation
        pass
```

### Using Fixtures

Common fixtures available in `conftest.py`:

- `test_data_dir`: Temporary directory for test data
- `sample_audio_file`: 1-second sine wave (440 Hz)
- `sample_audio_file_long`: 5-second multi-frequency audio
- `sample_stereo_audio_file`: Stereo audio file
- `temp_output_dir`: Temporary output directory
- `mock_gradio_progress`: Mock progress tracker
- `setup_test_directories`: Auto-setup test directories

### Writing Benchmark Tests

For performance benchmarks:

```python
@pytest.mark.benchmark
@pytest.mark.slow
class TestFeatureSpeed:
    """Benchmark feature performance."""
    
    def test_feature_speed(self, sample_audio_file, mock_gradio_progress):
        """Benchmark feature speed."""
        monitor = ResourceMonitor()
        monitor.start()
        
        start_time = time.time()
        result = feature_function(sample_audio_file, progress=mock_gradio_progress)
        elapsed = time.time() - start_time
        
        monitor.update()
        stats = monitor.get_stats()
        
        # Assert performance expectations
        assert elapsed < 5.0, f"Feature took {elapsed:.2f}s (expected < 5s)"
        
        # Print metrics
        print(f"\n📊 Feature Performance:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Memory: {stats['current_memory_mb']:.2f} MB")
```

## Continuous Integration

### GitHub Actions

The test suite is designed for CI environments. Example workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      - name: Run tests
        run: ./run_tests.sh fast
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

### Pre-commit Hooks

Configure pre-commit to run tests:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-fast
        name: pytest-fast
        entry: python -m pytest tests/ -v -m "not slow and not benchmark"
        language: system
        pass_filenames: false
        always_run: true
```

## Coverage Reports

### Generate Coverage Report

```bash
./run_tests.sh coverage
```

This generates:
- Terminal report (immediate feedback)
- HTML report (`htmlcov/index.html` - detailed view)
- XML report (`coverage.xml` - for CI tools)

### View HTML Coverage Report

```bash
# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Coverage Goals

- **Unit tests**: Aim for >80% coverage
- **Integration tests**: Focus on critical workflows
- **Overall project**: Target >70% coverage

### Coverage Configuration

Coverage is configured in `pytest.ini`:

```ini
[pytest]
addopts = 
    --cov=.
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'pytest'

```bash
pip install -r requirements-test.txt
```

#### Tests fail with missing dependencies

Install optional dependencies as needed:

```bash
# For MIDI tests
pip install basic-pitch

# For advanced separation
pip install audiosep

# For Demucs tests
pip install demucs
```

#### Slow test execution

Use parallel execution:

```bash
python -m pytest tests/ -n auto
```

Or skip slow tests:

```bash
./run_tests.sh fast
```

#### Memory errors during benchmarks

Reduce the number of tests or run them sequentially:

```bash
python -m pytest tests/integration/test_speed_benchmarks.py -v --maxfail=1
```

## Best Practices

1. **Run tests before committing**: `./run_tests.sh fast`
2. **Write tests for new features**: Maintain test coverage
3. **Use descriptive test names**: Tests serve as documentation
4. **Test edge cases**: Not just happy paths
5. **Keep tests independent**: No test should depend on another
6. **Use fixtures**: Avoid code duplication
7. **Mock external dependencies**: Keep tests fast and reliable
8. **Document complex tests**: Add docstrings explaining what's tested

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
- [Testing best practices](https://docs.python-guide.org/writing/tests/)

## Summary

The FORGE v1 test suite provides comprehensive coverage of:
- ✅ 47+ unit tests for core functionality
- ✅ 6+ integration tests for workflows
- ✅ 20+ performance benchmarks
- ✅ Speed, memory, and scalability tests
- ✅ Automated test runner script
- ✅ Coverage reporting
- ✅ CI/CD ready

Run `./run_tests.sh help` for a complete list of test commands.
