"""
Unit tests for audio processing functions in forgev1.py
"""
import pytest
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import sys
import os

# Add parent directory to path to import forgev1
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forgev1 import extract_loops, generate_vocal_chops


@pytest.mark.unit
class TestLoopExtraction:
    """Test loop extraction functionality."""
    
    def test_extract_loops_basic(self, sample_audio_file_long, mock_gradio_progress):
        """Test basic loop extraction."""
        loops = extract_loops(
            audio_path=sample_audio_file_long,
            loop_duration=2.0,
            aperture=0.5,
            num_loops=3,
            progress=mock_gradio_progress
        )
        
        assert isinstance(loops, list)
        assert len(loops) <= 3
        
        # Check loop structure
        for loop in loops:
            assert 'path' in loop
            assert 'start_time' in loop
            assert 'end_time' in loop
            assert 'score' in loop
            assert 'rank' in loop
            assert Path(loop['path']).exists()
            assert loop['end_time'] > loop['start_time']
            assert loop['score'] >= 0
    
    def test_extract_loops_aperture_range(self, sample_audio_file_long, mock_gradio_progress):
        """Test loop extraction with different aperture values."""
        # Test with energy-focused aperture (0.0)
        loops_energy = extract_loops(
            audio_path=sample_audio_file_long,
            loop_duration=2.0,
            aperture=0.0,
            num_loops=2,
            progress=mock_gradio_progress
        )
        assert len(loops_energy) > 0
        
        # Test with spectral-focused aperture (1.0)
        loops_spectral = extract_loops(
            audio_path=sample_audio_file_long,
            loop_duration=2.0,
            aperture=1.0,
            num_loops=2,
            progress=mock_gradio_progress
        )
        assert len(loops_spectral) > 0
        
        # Scores should potentially differ based on aperture
        # (though with synthetic audio they might be similar)
    
    def test_extract_loops_sorted_by_score(self, sample_audio_file_long, mock_gradio_progress):
        """Test that loops are sorted by score."""
        loops = extract_loops(
            audio_path=sample_audio_file_long,
            loop_duration=2.0,
            aperture=0.5,
            num_loops=5,
            progress=mock_gradio_progress
        )
        
        # Check that scores are in descending order
        scores = [loop['score'] for loop in loops]
        assert scores == sorted(scores, reverse=True)
    
    def test_extract_loops_invalid_file(self, mock_gradio_progress):
        """Test loop extraction with invalid file."""
        with pytest.raises(Exception):
            extract_loops(
                audio_path="nonexistent_file.wav",
                loop_duration=2.0,
                aperture=0.5,
                num_loops=3,
                progress=mock_gradio_progress
            )


@pytest.mark.unit
class TestVocalChops:
    """Test vocal chop generation."""
    
    def test_generate_chops_onset_mode(self, sample_audio_file_long, mock_gradio_progress):
        """Test vocal chop generation with onset detection."""
        chops = generate_vocal_chops(
            audio_path=sample_audio_file_long,
            mode='onset',
            min_duration=0.1,
            max_duration=2.0,
            threshold=0.3,
            progress=mock_gradio_progress
        )
        
        assert isinstance(chops, list)
        # With synthetic audio, we might get chops
        for chop_path in chops:
            assert Path(chop_path).exists()
            # Verify it's a valid audio file
            y, sr = librosa.load(chop_path, sr=None)
            assert len(y) > 0
    
    def test_generate_chops_silence_mode(self, sample_audio_file_long, mock_gradio_progress):
        """Test vocal chop generation with silence detection."""
        chops = generate_vocal_chops(
            audio_path=sample_audio_file_long,
            mode='silence',
            min_duration=0.1,
            max_duration=2.0,
            threshold=0.3,
            progress=mock_gradio_progress
        )
        
        assert isinstance(chops, list)
    
    def test_generate_chops_hybrid_mode(self, sample_audio_file_long, mock_gradio_progress):
        """Test vocal chop generation with hybrid detection."""
        chops = generate_vocal_chops(
            audio_path=sample_audio_file_long,
            mode='hybrid',
            min_duration=0.1,
            max_duration=2.0,
            threshold=0.3,
            progress=mock_gradio_progress
        )
        
        assert isinstance(chops, list)
    
    def test_generate_chops_duration_constraints(self, sample_audio_file_long, mock_gradio_progress):
        """Test that chops respect duration constraints."""
        min_dur = 0.2
        max_dur = 1.0
        
        chops = generate_vocal_chops(
            audio_path=sample_audio_file_long,
            mode='onset',
            min_duration=min_dur,
            max_duration=max_dur,
            threshold=0.3,
            progress=mock_gradio_progress
        )
        
        # Check each chop's duration
        for chop_path in chops:
            y, sr = librosa.load(chop_path, sr=None)
            duration = len(y) / sr
            # Allow small tolerance for edge effects
            assert duration >= min_dur - 0.05
            assert duration <= max_dur + 0.05
