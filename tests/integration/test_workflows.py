"""
Integration tests for complete workflows in app.py
"""
import pytest
from pathlib import Path
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    extract_loops,
    generate_vocal_chops,
    save_feedback,
    setup_directories
)


@pytest.mark.integration
class TestCompleteWorkflows:
    """Test complete processing workflows."""
    
    def test_end_to_end_loop_workflow(self, sample_audio_file_long, mock_gradio_progress):
        """Test complete loop extraction workflow."""
        # Setup
        setup_directories()
        
        # Extract loops
        loops = extract_loops(
            audio_path=sample_audio_file_long,
            loop_duration=2.0,
            aperture=0.5,
            num_loops=3,
            progress=mock_gradio_progress
        )
        
        # Verify results
        assert len(loops) > 0
        assert all(Path(loop['path']).exists() for loop in loops)
        
        # Verify output directory structure
        assert Path('output/loops').exists()
    
    def test_end_to_end_chop_workflow(self, sample_audio_file_long, mock_gradio_progress):
        """Test complete vocal chop workflow."""
        # Setup
        setup_directories()
        
        # Generate chops
        chops = generate_vocal_chops(
            audio_path=sample_audio_file_long,
            mode='onset',
            min_duration=0.1,
            max_duration=2.0,
            threshold=0.3,
            progress=mock_gradio_progress
        )
        
        # Verify output directory
        assert Path('output/chops').exists()
    
    def test_feedback_workflow(self):
        """Test feedback submission workflow."""
        # Setup
        setup_directories()
        
        # Submit feedback
        result = save_feedback(
            feature="Loop Extraction",
            rating=5,
            comments="Great feature!",
            email="test@example.com"
        )
        
        # Verify feedback was saved
        assert "saved successfully" in result.lower()
        assert Path('feedback').exists()
        
        # Check that a feedback file was created
        feedback_files = list(Path('feedback').glob('*.json'))
        assert len(feedback_files) > 0
    
    def test_multiple_operations_sequence(self, sample_audio_file_long, mock_gradio_progress):
        """Test running multiple operations in sequence."""
        setup_directories()
        
        # First extract loops
        loops = extract_loops(
            audio_path=sample_audio_file_long,
            loop_duration=2.0,
            aperture=0.5,
            num_loops=2,
            progress=mock_gradio_progress
        )
        assert len(loops) > 0
        
        # Then generate chops from same audio
        chops = generate_vocal_chops(
            audio_path=sample_audio_file_long,
            mode='onset',
            min_duration=0.1,
            max_duration=2.0,
            threshold=0.3,
            progress=mock_gradio_progress
        )
        
        # Verify both outputs exist
        assert Path('output/loops').exists()
        assert Path('output/chops').exists()


@pytest.mark.integration
class TestDirectoryManagement:
    """Test directory setup and management."""
    
    def test_directory_creation(self):
        """Test that all required directories are created."""
        dirs = setup_directories()
        
        required_dirs = [
            'runs', 'output', 'cache', 'config', 'checkpoint', 'feedback',
            'output/stems', 'output/loops', 'output/chops',
            'output/midi', 'output/drums', 'output/videos'
        ]
        
        for directory in required_dirs:
            assert Path(directory).exists()
            assert Path(directory).is_dir()
    
    def test_repeated_directory_setup(self):
        """Test that directory setup can be called multiple times safely."""
        setup_directories()
        setup_directories()  # Should not raise error
        
        assert Path('output').exists()
