"""
Pytest configuration and fixtures for the test suite.
"""
import os
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="forge_test_")
    yield Path(temp_dir)
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_audio_file(test_data_dir):
    """
    Generate a simple test audio file (1 second sine wave at 440 Hz).
    """
    sample_rate = 44100
    duration = 1.0  # seconds
    frequency = 440.0  # Hz (A4 note)
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Save to file
    audio_path = test_data_dir / "test_audio.wav"
    sf.write(audio_path, audio, sample_rate)
    
    return str(audio_path)


@pytest.fixture(scope="session")
def sample_audio_file_long(test_data_dir):
    """
    Generate a longer test audio file (5 seconds) for loop extraction tests.
    """
    sample_rate = 44100
    duration = 5.0  # seconds
    
    # Generate a more complex waveform with multiple frequencies
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (0.3 * np.sin(2 * np.pi * 220 * t) +  # A3
             0.3 * np.sin(2 * np.pi * 330 * t) +  # E4
             0.2 * np.sin(2 * np.pi * 440 * t))   # A4
    
    # Add some rhythmic variation
    envelope = np.concatenate([
        np.linspace(0, 1, sample_rate // 4),
        np.ones(sample_rate // 4),
        np.linspace(1, 0.3, sample_rate // 2)
    ])
    envelope = np.tile(envelope, int(duration) + 1)[:len(audio)]
    audio = audio * envelope
    
    # Save to file
    audio_path = test_data_dir / "test_audio_long.wav"
    sf.write(audio_path, audio, sample_rate)
    
    return str(audio_path)


@pytest.fixture(scope="session")
def sample_stereo_audio_file(test_data_dir):
    """
    Generate a stereo test audio file.
    """
    sample_rate = 44100
    duration = 1.0
    
    # Generate stereo audio (different frequencies in each channel)
    t = np.linspace(0, duration, int(sample_rate * duration))
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 554.37 * t)  # C#5
    
    audio = np.vstack([left, right]).T
    
    # Save to file
    audio_path = test_data_dir / "test_audio_stereo.wav"
    sf.write(audio_path, audio, sample_rate)
    
    return str(audio_path)


@pytest.fixture
def temp_output_dir(test_data_dir):
    """Create a temporary output directory for each test."""
    output_dir = test_data_dir / "output"
    output_dir.mkdir(exist_ok=True)
    yield output_dir
    # Cleanup after each test
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture
def mock_gradio_progress():
    """Mock Gradio progress tracker for testing."""
    class MockProgress:
        def __call__(self, value, desc=""):
            pass
    return MockProgress()


@pytest.fixture(autouse=True)
def setup_test_directories(tmp_path, monkeypatch):
    """
    Setup test directories for each test.
    This fixture automatically runs before each test.
    """
    # Create test directories
    test_dirs = [
        'runs', 'output', 'cache', 'config', 'checkpoint', 'feedback',
        'output/stems', 'output/loops', 'output/chops',
        'output/midi', 'output/drums', 'output/videos'
    ]
    
    for dir_name in test_dirs:
        (tmp_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Change working directory to temp path for isolation
    monkeypatch.chdir(tmp_path)
    
    return tmp_path
