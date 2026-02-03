#!/usr/bin/env python3
"""
Batch Processing Module for FORGE v1
====================================

Provides batch processing capabilities for all core operations:
- Batch stem separation
- Batch loop extraction  
- Batch vocal chop generation
- Batch MIDI extraction
- Batch drum one-shot generation

Author: NeuralWorkstation Team
License: MIT
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import gradio as gr

# Import core functions from forgev1
from forgev1 import (
    separate_stems_demucs,
    extract_loops,
    generate_vocal_chops,
    extract_midi,
    generate_drum_oneshots,
    setup_directories
)


class BatchProcessor:
    """
    Handles batch processing operations with progress tracking and error handling.
    """
    
    def __init__(self, max_workers: int = 2):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum number of parallel workers (default: 2 for safety)
        """
        self.max_workers = max_workers
        self.results = []
        self.errors = []
    
    def process_batch(
        self,
        files: List[str],
        operation: Callable,
        operation_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process multiple files with a given operation.
        
        Args:
            files: List of file paths to process
            operation: Function to call for each file
            operation_name: Name of the operation for logging
            **kwargs: Additional arguments to pass to the operation
            
        Returns:
            Dictionary with results and errors
        """
        self.results = []
        self.errors = []
        
        total_files = len(files)
        processed = 0
        
        print(f"\n🔄 Starting batch {operation_name} for {total_files} files...")
        
        for file_path in files:
            try:
                print(f"  Processing: {Path(file_path).name}")
                result = operation(audio_path=file_path, **kwargs)
                self.results.append({
                    'file': file_path,
                    'success': True,
                    'result': result
                })
                processed += 1
                print(f"  ✅ Success ({processed}/{total_files})")
                
            except Exception as e:
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                self.errors.append({
                    'file': file_path,
                    'error': error_msg
                })
                print(f"  ❌ Error: {str(e)}")
        
        success_count = len(self.results)
        error_count = len(self.errors)
        
        summary = {
            'total': total_files,
            'success': success_count,
            'errors': error_count,
            'results': self.results,
            'error_details': self.errors
        }
        
        print(f"\n✅ Batch complete: {success_count} succeeded, {error_count} failed")
        
        return summary
    
    def save_batch_report(self, summary: Dict[str, Any], operation_name: str) -> str:
        """
        Save batch processing report to JSON file.
        
        Args:
            summary: Batch processing summary
            operation_name: Name of the operation
            
        Returns:
            Path to saved report file
        """
        # Create reports directory
        reports_dir = Path('output/batch_reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = reports_dir / f"batch_{operation_name}_{timestamp}.json"
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📄 Report saved: {report_path}")
        return str(report_path)


def batch_separate_stems(
    files: List[str],
    model: str = 'htdemucs',
    use_cache: bool = True,
    progress=gr.Progress()
) -> str:
    """
    Batch stem separation using Demucs.
    
    Args:
        files: List of audio file paths
        model: Demucs model to use
        use_cache: Whether to use cached results
        progress: Gradio progress tracker
        
    Returns:
        Summary message with results
    """
    if not files:
        return "❌ No files provided"
    
    progress(0, desc=f"Starting batch stem separation for {len(files)} files...")
    
    processor = BatchProcessor(max_workers=1)  # Demucs is heavy, use 1 worker
    
    # Create mock progress for individual operations
    mock_progress = type('MockProgress', (), {'__call__': lambda self, *args, **kwargs: None})()
    
    summary = processor.process_batch(
        files=files,
        operation=separate_stems_demucs,
        operation_name="stem_separation",
        model=model,
        use_cache=use_cache,
        progress=mock_progress
    )
    
    report_path = processor.save_batch_report(summary, "stem_separation")
    
    progress(1.0, desc="Batch processing complete!")
    
    return f"""
    ✅ Batch Stem Separation Complete
    
    📊 Results:
    - Total files: {summary['total']}
    - Successful: {summary['success']}
    - Failed: {summary['errors']}
    
    📄 Report: {report_path}
    """


def batch_extract_loops(
    files: List[str],
    loop_duration: float = 4.0,
    aperture: float = 0.5,
    num_loops: int = 5,
    progress=gr.Progress()
) -> str:
    """
    Batch loop extraction from multiple audio files.
    
    Args:
        files: List of audio file paths
        loop_duration: Duration of each loop in seconds
        aperture: Aperture control (0-1)
        num_loops: Number of loops to extract per file
        progress: Gradio progress tracker
        
    Returns:
        Summary message with results
    """
    if not files:
        return "❌ No files provided"
    
    progress(0, desc=f"Starting batch loop extraction for {len(files)} files...")
    
    processor = BatchProcessor(max_workers=2)
    mock_progress = type('MockProgress', (), {'__call__': lambda self, *args, **kwargs: None})()
    
    summary = processor.process_batch(
        files=files,
        operation=extract_loops,
        operation_name="loop_extraction",
        loop_duration=loop_duration,
        aperture=aperture,
        num_loops=num_loops,
        progress=mock_progress
    )
    
    report_path = processor.save_batch_report(summary, "loop_extraction")
    
    progress(1.0, desc="Batch processing complete!")
    
    return f"""
    ✅ Batch Loop Extraction Complete
    
    📊 Results:
    - Total files: {summary['total']}
    - Successful: {summary['success']}
    - Failed: {summary['errors']}
    - Loops extracted: {sum(len(r['result']) for r in summary['results'])}
    
    📄 Report: {report_path}
    """


def batch_generate_chops(
    files: List[str],
    mode: str = 'onset',
    min_duration: float = 0.1,
    max_duration: float = 2.0,
    threshold: float = 0.3,
    progress=gr.Progress()
) -> str:
    """
    Batch vocal chop generation from multiple audio files.
    
    Args:
        files: List of audio file paths
        mode: Detection mode ('silence', 'onset', 'hybrid')
        min_duration: Minimum chop duration
        max_duration: Maximum chop duration
        threshold: Detection threshold
        progress: Gradio progress tracker
        
    Returns:
        Summary message with results
    """
    if not files:
        return "❌ No files provided"
    
    progress(0, desc=f"Starting batch chop generation for {len(files)} files...")
    
    processor = BatchProcessor(max_workers=2)
    mock_progress = type('MockProgress', (), {'__call__': lambda self, *args, **kwargs: None})()
    
    summary = processor.process_batch(
        files=files,
        operation=generate_vocal_chops,
        operation_name="chop_generation",
        mode=mode,
        min_duration=min_duration,
        max_duration=max_duration,
        threshold=threshold,
        progress=mock_progress
    )
    
    report_path = processor.save_batch_report(summary, "chop_generation")
    
    progress(1.0, desc="Batch processing complete!")
    
    return f"""
    ✅ Batch Chop Generation Complete
    
    📊 Results:
    - Total files: {summary['total']}
    - Successful: {summary['success']}
    - Failed: {summary['errors']}
    - Chops created: {sum(len(r['result']) for r in summary['results'])}
    
    📄 Report: {report_path}
    """


def batch_extract_midi(
    files: List[str],
    progress=gr.Progress()
) -> str:
    """
    Batch MIDI extraction from multiple audio files.
    
    Args:
        files: List of audio file paths
        progress: Gradio progress tracker
        
    Returns:
        Summary message with results
    """
    if not files:
        return "❌ No files provided"
    
    progress(0, desc=f"Starting batch MIDI extraction for {len(files)} files...")
    
    processor = BatchProcessor(max_workers=2)
    mock_progress = type('MockProgress', (), {'__call__': lambda self, *args, **kwargs: None})()
    
    summary = processor.process_batch(
        files=files,
        operation=extract_midi,
        operation_name="midi_extraction",
        progress=mock_progress
    )
    
    report_path = processor.save_batch_report(summary, "midi_extraction")
    
    progress(1.0, desc="Batch processing complete!")
    
    return f"""
    ✅ Batch MIDI Extraction Complete
    
    📊 Results:
    - Total files: {summary['total']}
    - Successful: {summary['success']}
    - Failed: {summary['errors']}
    
    📄 Report: {report_path}
    """


def batch_generate_drum_oneshots(
    files: List[str],
    min_duration: float = 0.05,
    max_duration: float = 0.5,
    apply_fadeout: bool = True,
    progress=gr.Progress()
) -> str:
    """
    Batch drum one-shot generation from multiple audio files.
    
    Args:
        files: List of audio file paths
        min_duration: Minimum one-shot duration
        max_duration: Maximum one-shot duration
        apply_fadeout: Whether to apply fade-out
        progress: Gradio progress tracker
        
    Returns:
        Summary message with results
    """
    if not files:
        return "❌ No files provided"
    
    progress(0, desc=f"Starting batch drum one-shot generation for {len(files)} files...")
    
    processor = BatchProcessor(max_workers=2)
    mock_progress = type('MockProgress', (), {'__call__': lambda self, *args, **kwargs: None})()
    
    summary = processor.process_batch(
        files=files,
        operation=generate_drum_oneshots,
        operation_name="drum_oneshot_generation",
        min_duration=min_duration,
        max_duration=max_duration,
        apply_fadeout=apply_fadeout,
        progress=mock_progress
    )
    
    report_path = processor.save_batch_report(summary, "drum_oneshot_generation")
    
    progress(1.0, desc="Batch processing complete!")
    
    return f"""
    ✅ Batch Drum One-Shot Generation Complete
    
    📊 Results:
    - Total files: {summary['total']}
    - Successful: {summary['success']}
    - Failed: {summary['errors']}
    - One-shots created: {sum(len(r['result']) for r in summary['results'])}
    
    📄 Report: {report_path}
    """
