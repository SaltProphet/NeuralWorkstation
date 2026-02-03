"""
Unit tests for batch processing functionality
"""
import pytest
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_processor import (
    BatchProcessor,
    batch_extract_loops,
    batch_generate_chops
)


@pytest.mark.unit
class TestBatchProcessor:
    """Test BatchProcessor class."""
    
    def test_batch_processor_initialization(self):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor(max_workers=2)
        assert processor.max_workers == 2
        assert processor.results == []
        assert processor.errors == []
    
    def test_save_batch_report(self, tmp_path, monkeypatch):
        """Test saving batch report."""
        monkeypatch.chdir(tmp_path)
        
        processor = BatchProcessor()
        summary = {
            'total': 2,
            'success': 2,
            'errors': 0,
            'results': [],
            'error_details': []
        }
        
        report_path = processor.save_batch_report(summary, "test_operation")
        assert Path(report_path).exists()
        assert "batch_test_operation" in report_path


@pytest.mark.integration
class TestBatchOperations:
    """Test batch operations."""
    
    def test_batch_extract_loops(self, sample_audio_file_long, sample_audio_file, mock_gradio_progress):
        """Test batch loop extraction."""
        files = [sample_audio_file_long, sample_audio_file]
        
        result = batch_extract_loops(
            files=files,
            loop_duration=2.0,
            aperture=0.5,
            num_loops=2,
            progress=mock_gradio_progress
        )
        
        assert isinstance(result, str)
        assert "Batch Loop Extraction Complete" in result
        assert "Total files: 2" in result
    
    def test_batch_generate_chops(self, sample_audio_file_long, sample_audio_file, mock_gradio_progress):
        """Test batch chop generation."""
        files = [sample_audio_file_long, sample_audio_file]
        
        result = batch_generate_chops(
            files=files,
            mode='onset',
            min_duration=0.1,
            max_duration=2.0,
            threshold=0.3,
            progress=mock_gradio_progress
        )
        
        assert isinstance(result, str)
        assert "Batch Chop Generation Complete" in result
        assert "Total files: 2" in result
    
    def test_batch_empty_files(self, mock_gradio_progress):
        """Test batch operations with empty file list."""
        result = batch_extract_loops(
            files=[],
            loop_duration=2.0,
            aperture=0.5,
            num_loops=2,
            progress=mock_gradio_progress
        )
        
        assert "No files provided" in result
    
    def test_batch_report_generation(self, sample_audio_file, mock_gradio_progress):
        """Test that batch reports are generated."""
        files = [sample_audio_file]
        
        result = batch_extract_loops(
            files=files,
            loop_duration=2.0,
            aperture=0.5,
            num_loops=2,
            progress=mock_gradio_progress
        )
        
        # Check that report path is in result
        assert "Report:" in result
        
        # Check that report directory was created
        assert Path('output/batch_reports').exists()
