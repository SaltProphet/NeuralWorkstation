"""
Unit tests for utility functions in app.py
"""
import pytest
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    sanitize_filename,
    get_audio_hash,
    format_timestamp,
    db_to_amplitude,
    amplitude_to_db,
    setup_directories,
    Config
)


@pytest.mark.unit
class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Test basic sanitization
        assert sanitize_filename("normal_file.wav") == "normal_file_wav"
        
        # Test special character removal
        assert sanitize_filename("file@#$%name.wav") == "file_name_wav"
        
        # Test path traversal prevention
        assert sanitize_filename("../../etc/passwd") == "etc_passwd"
        
        # Test multiple spaces/underscores
        assert sanitize_filename("file    name___test") == "file_name_test"
        
        # Test leading/trailing underscores
        assert sanitize_filename("___file___") == "file"
        
        # Test length limit
        long_name = "a" * 200
        assert len(sanitize_filename(long_name)) == 100
    
    def test_get_audio_hash(self, sample_audio_file):
        """Test audio file hashing."""
        hash1 = get_audio_hash(sample_audio_file)
        hash2 = get_audio_hash(sample_audio_file)
        
        # Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
        assert hash1.isalnum()  # Should be alphanumeric
    
    def test_format_timestamp(self):
        """Test timestamp formatting."""
        assert format_timestamp(0) == "00:00.000"
        assert format_timestamp(61.5) == "01:01.500"
        assert format_timestamp(125.123) == "02:05.123"
        assert format_timestamp(3599.999) == "59:59.999"
    
    def test_db_to_amplitude(self):
        """Test decibel to amplitude conversion."""
        assert db_to_amplitude(0) == pytest.approx(1.0)
        assert db_to_amplitude(-6) == pytest.approx(0.501187, rel=1e-5)
        assert db_to_amplitude(-20) == pytest.approx(0.1)
        assert db_to_amplitude(6) == pytest.approx(1.995262, rel=1e-5)
    
    def test_amplitude_to_db(self):
        """Test amplitude to decibel conversion."""
        assert amplitude_to_db(1.0) == pytest.approx(0.0)
        assert amplitude_to_db(0.5) == pytest.approx(-6.0206, rel=1e-3)
        assert amplitude_to_db(0.1) == pytest.approx(-20.0)
        assert amplitude_to_db(2.0) == pytest.approx(6.0206, rel=1e-3)
        
        # Test handling of very small values
        assert amplitude_to_db(0.0) < -100  # Should not crash
    
    def test_db_amplitude_roundtrip(self):
        """Test that db <-> amplitude conversions are reversible."""
        test_values = [0.1, 0.5, 1.0, 2.0, 10.0]
        for value in test_values:
            db = amplitude_to_db(value)
            recovered = db_to_amplitude(db)
            assert recovered == pytest.approx(value, rel=1e-9)
    
    def test_setup_directories(self):
        """Test directory setup."""
        dirs = setup_directories()
        
        # Check that all directories were created
        assert len(dirs) > 0
        for directory in dirs:
            assert Path(directory).exists()
            assert Path(directory).is_dir()
    
    def test_config_save_load(self, tmp_path):
        """Test configuration save and load."""
        test_config = {
            'sample_rate': 48000,
            'model': 'htdemucs_ft',
            'cache_enabled': True
        }
        
        Config.save_config(test_config, 'test_config')
        loaded_config = Config.load_config('test_config')
        
        assert loaded_config == test_config
    
    def test_config_default_values(self):
        """Test that Config has expected default values."""
        assert Config.SAMPLE_RATE == 44100
        assert Config.HOP_LENGTH == 512
        assert Config.N_FFT == 2048
        assert Config.DEFAULT_LOOP_LENGTH == 4
        assert len(Config.DEMUCS_MODELS) > 0
        assert 'htdemucs' in Config.DEMUCS_MODELS
