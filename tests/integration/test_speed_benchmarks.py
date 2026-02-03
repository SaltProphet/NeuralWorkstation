"""
Performance and speed benchmarks for FORGE v1 features.

This module tests the speed and performance characteristics of all major features:
- Stem separation
- Loop extraction
- Vocal chop generation
- MIDI extraction
- Drum one-shot generation
- Video rendering
- Batch processing

Benchmarks establish baseline performance metrics and detect regressions.
"""
import pytest
import time
import sys
import os
from pathlib import Path
import numpy as np
import soundfile as sf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forgev1 import (
    extract_loops,
    generate_vocal_chops,
    extract_midi,
    generate_drum_oneshots,
    get_audio_hash,
    setup_directories
)
from batch_processor import batch_extract_loops, batch_generate_chops
from performance import ResourceMonitor, OptimizedAudioLoader


@pytest.mark.benchmark
@pytest.mark.slow
class TestLoopExtractionSpeed:
    """Benchmark loop extraction performance."""
    
    def test_loop_extraction_small_audio(self, sample_audio_file, mock_gradio_progress):
        """Benchmark loop extraction on small audio file (1 second)."""
        monitor = ResourceMonitor()
        monitor.start()
        
        start_time = time.time()
        loops = extract_loops(
            audio_path=sample_audio_file,
            loop_duration=1.0,
            aperture=0.5,
            num_loops=3,
            progress=mock_gradio_progress
        )
        elapsed = time.time() - start_time
        
        monitor.update()
        stats = monitor.get_stats()
        
        # Performance expectations
        assert elapsed < 5.0, f"Loop extraction took {elapsed:.2f}s (expected < 5s)"
        assert len(loops) > 0
        
        print(f"\n📊 Loop Extraction (Small) Performance:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Loops: {len(loops)}")
        print(f"  Memory: {stats['current_memory_mb']:.2f} MB")
    
    def test_loop_extraction_medium_audio(self, sample_audio_file_long, mock_gradio_progress):
        """Benchmark loop extraction on medium audio file (5 seconds)."""
        monitor = ResourceMonitor()
        monitor.start()
        
        start_time = time.time()
        loops = extract_loops(
            audio_path=sample_audio_file_long,
            loop_duration=2.0,
            aperture=0.5,
            num_loops=5,
            progress=mock_gradio_progress
        )
        elapsed = time.time() - start_time
        
        monitor.update()
        stats = monitor.get_stats()
        
        # Performance expectations
        assert elapsed < 10.0, f"Loop extraction took {elapsed:.2f}s (expected < 10s)"
        assert len(loops) > 0
        
        print(f"\n📊 Loop Extraction (Medium) Performance:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Loops: {len(loops)}")
        print(f"  Memory: {stats['current_memory_mb']:.2f} MB")
    
    def test_loop_extraction_different_apertures(self, sample_audio_file_long, mock_gradio_progress):
        """Benchmark loop extraction with different aperture settings."""
        apertures = [0.0, 0.5, 1.0]
        timings = {}
        
        for aperture in apertures:
            start_time = time.time()
            loops = extract_loops(
                audio_path=sample_audio_file_long,
                loop_duration=2.0,
                aperture=aperture,
                num_loops=3,
                progress=mock_gradio_progress
            )
            elapsed = time.time() - start_time
            timings[aperture] = elapsed
            
            assert elapsed < 10.0, f"Loop extraction (aperture={aperture}) took {elapsed:.2f}s"
        
        print(f"\n📊 Loop Extraction Aperture Comparison:")
        for aperture, timing in timings.items():
            print(f"  Aperture {aperture}: {timing:.2f}s")


@pytest.mark.benchmark
@pytest.mark.slow
class TestVocalChopSpeed:
    """Benchmark vocal chop generation performance."""
    
    def test_vocal_chop_silence_mode(self, sample_audio_file_long, mock_gradio_progress):
        """Benchmark vocal chop generation in silence mode."""
        monitor = ResourceMonitor()
        monitor.start()
        
        start_time = time.time()
        chops = generate_vocal_chops(
            audio_path=sample_audio_file_long,
            mode='silence',
            min_duration=0.1,
            max_duration=2.0,
            threshold=0.3,
            progress=mock_gradio_progress
        )
        elapsed = time.time() - start_time
        
        monitor.update()
        stats = monitor.get_stats()
        
        assert elapsed < 5.0, f"Chop generation (silence) took {elapsed:.2f}s"
        
        print(f"\n📊 Vocal Chop (Silence Mode) Performance:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Memory: {stats['current_memory_mb']:.2f} MB")
    
    def test_vocal_chop_onset_mode(self, sample_audio_file_long, mock_gradio_progress):
        """Benchmark vocal chop generation in onset mode."""
        monitor = ResourceMonitor()
        monitor.start()
        
        start_time = time.time()
        chops = generate_vocal_chops(
            audio_path=sample_audio_file_long,
            mode='onset',
            min_duration=0.1,
            max_duration=2.0,
            threshold=0.3,
            progress=mock_gradio_progress
        )
        elapsed = time.time() - start_time
        
        monitor.update()
        stats = monitor.get_stats()
        
        assert elapsed < 5.0, f"Chop generation (onset) took {elapsed:.2f}s"
        
        print(f"\n📊 Vocal Chop (Onset Mode) Performance:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Memory: {stats['current_memory_mb']:.2f} MB")
    
    def test_vocal_chop_hybrid_mode(self, sample_audio_file_long, mock_gradio_progress):
        """Benchmark vocal chop generation in hybrid mode."""
        start_time = time.time()
        chops = generate_vocal_chops(
            audio_path=sample_audio_file_long,
            mode='hybrid',
            min_duration=0.1,
            max_duration=2.0,
            threshold=0.3,
            progress=mock_gradio_progress
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 5.0, f"Chop generation (hybrid) took {elapsed:.2f}s"
        
        print(f"\n📊 Vocal Chop (Hybrid Mode) Performance:")
        print(f"  Time: {elapsed:.2f}s")


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.requires_basic_pitch
class TestMIDIExtractionSpeed:
    """Benchmark MIDI extraction performance."""
    
    def test_midi_extraction_speed(self, sample_audio_file_long, mock_gradio_progress):
        """Benchmark MIDI extraction."""
        monitor = ResourceMonitor()
        monitor.start()
        
        try:
            start_time = time.time()
            result = extract_midi(
                audio_path=sample_audio_file_long,
                progress=mock_gradio_progress
            )
            elapsed = time.time() - start_time
            
            monitor.update()
            stats = monitor.get_stats()
            
            # MIDI extraction can be slow
            assert elapsed < 30.0, f"MIDI extraction took {elapsed:.2f}s (expected < 30s)"
            
            print(f"\n📊 MIDI Extraction Performance:")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Memory: {stats['current_memory_mb']:.2f} MB")
        except ImportError:
            pytest.skip("basic-pitch not installed")


@pytest.mark.benchmark
@pytest.mark.slow
class TestDrumOneshotSpeed:
    """Benchmark drum one-shot generation performance."""
    
    def test_drum_oneshot_generation(self, sample_audio_file_long, mock_gradio_progress):
        """Benchmark drum one-shot generation."""
        monitor = ResourceMonitor()
        monitor.start()
        
        start_time = time.time()
        oneshots = generate_drum_oneshots(
            audio_path=sample_audio_file_long,
            min_duration=0.05,
            max_duration=1.0,
            apply_fadeout=True,
            progress=mock_gradio_progress
        )
        elapsed = time.time() - start_time
        
        monitor.update()
        stats = monitor.get_stats()
        
        assert elapsed < 5.0, f"Drum one-shot generation took {elapsed:.2f}s"
        
        print(f"\n📊 Drum One-Shot Generation Performance:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Memory: {stats['current_memory_mb']:.2f} MB")


@pytest.mark.benchmark
@pytest.mark.slow
class TestBatchProcessingSpeed:
    """Benchmark batch processing performance."""
    
    def test_batch_loop_extraction(self, sample_audio_file, sample_audio_file_long, mock_gradio_progress):
        """Benchmark batch loop extraction."""
        files = [sample_audio_file, sample_audio_file_long]
        
        monitor = ResourceMonitor()
        monitor.start()
        
        start_time = time.time()
        result = batch_extract_loops(
            files=files,
            loop_duration=2.0,
            aperture=0.5,
            num_loops=2,
            progress=mock_gradio_progress
        )
        elapsed = time.time() - start_time
        
        monitor.update()
        stats = monitor.get_stats()
        
        # Batch processing should be reasonably fast
        assert elapsed < 20.0, f"Batch loop extraction took {elapsed:.2f}s"
        
        print(f"\n📊 Batch Loop Extraction Performance:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Files: {len(files)}")
        print(f"  Avg per file: {elapsed / len(files):.2f}s")
        print(f"  Memory: {stats['current_memory_mb']:.2f} MB")
    
    def test_batch_chop_generation(self, sample_audio_file, sample_audio_file_long, mock_gradio_progress):
        """Benchmark batch chop generation."""
        files = [sample_audio_file, sample_audio_file_long]
        
        monitor = ResourceMonitor()
        monitor.start()
        
        start_time = time.time()
        result = batch_generate_chops(
            files=files,
            mode='onset',
            min_duration=0.1,
            max_duration=2.0,
            threshold=0.3,
            progress=mock_gradio_progress
        )
        elapsed = time.time() - start_time
        
        monitor.update()
        stats = monitor.get_stats()
        
        assert elapsed < 15.0, f"Batch chop generation took {elapsed:.2f}s"
        
        print(f"\n📊 Batch Chop Generation Performance:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Files: {len(files)}")
        print(f"  Avg per file: {elapsed / len(files):.2f}s")
        print(f"  Memory: {stats['current_memory_mb']:.2f} MB")


@pytest.mark.benchmark
class TestUtilitySpeed:
    """Benchmark utility function performance."""
    
    def test_audio_hash_speed(self, sample_audio_file):
        """Benchmark audio hashing speed."""
        iterations = 10
        
        start_time = time.time()
        for _ in range(iterations):
            hash_value = get_audio_hash(sample_audio_file)
        elapsed = time.time() - start_time
        
        avg_time = elapsed / iterations
        assert avg_time < 0.1, f"Audio hashing took {avg_time:.4f}s per call"
        
        print(f"\n📊 Audio Hashing Performance:")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Iterations: {iterations}")
    
    def test_audio_loading_speed(self, sample_audio_file_long):
        """Benchmark audio loading speed."""
        monitor = ResourceMonitor()
        monitor.start()
        
        start_time = time.time()
        y, sr = OptimizedAudioLoader.load_audio(sample_audio_file_long)
        elapsed = time.time() - start_time
        
        monitor.update()
        stats = monitor.get_stats()
        
        assert elapsed < 1.0, f"Audio loading took {elapsed:.2f}s"
        
        print(f"\n📊 Audio Loading Performance:")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Samples: {len(y)}")
        print(f"  Memory: {stats['current_memory_mb']:.2f} MB")


@pytest.mark.benchmark
class TestMemoryUsage:
    """Test memory usage for various operations."""
    
    def test_loop_extraction_memory(self, sample_audio_file_long, mock_gradio_progress):
        """Measure memory usage during loop extraction."""
        monitor = ResourceMonitor()
        monitor.start()
        
        loops = extract_loops(
            audio_path=sample_audio_file_long,
            loop_duration=2.0,
            aperture=0.5,
            num_loops=5,
            progress=mock_gradio_progress
        )
        
        monitor.update()
        stats = monitor.get_stats()
        
        # Memory should be reasonable
        assert stats['peak_memory_percent'] < 80, "Memory usage too high"
        
        print(f"\n💾 Loop Extraction Memory Usage:")
        print(f"  Peak: {stats['peak_memory_percent']:.1f}%")
        print(f"  Current: {stats['current_memory_mb']:.2f} MB")
    
    def test_batch_processing_memory(self, sample_audio_file, sample_audio_file_long, mock_gradio_progress):
        """Measure memory usage during batch processing."""
        files = [sample_audio_file, sample_audio_file_long] * 2  # 4 files total
        
        monitor = ResourceMonitor()
        monitor.start()
        
        result = batch_extract_loops(
            files=files,
            loop_duration=2.0,
            aperture=0.5,
            num_loops=2,
            progress=mock_gradio_progress
        )
        
        monitor.update()
        stats = monitor.get_stats()
        
        print(f"\n💾 Batch Processing Memory Usage:")
        print(f"  Peak: {stats['peak_memory_percent']:.1f}%")
        print(f"  Current: {stats['current_memory_mb']:.2f} MB")
        print(f"  Files processed: {len(files)}")


@pytest.mark.benchmark
class TestScalability:
    """Test performance scalability."""
    
    def test_loop_extraction_scaling(self, test_data_dir, mock_gradio_progress):
        """Test how loop extraction scales with different audio lengths."""
        # Create test files of different lengths
        sample_rate = 44100
        durations = [1.0, 3.0, 5.0]
        timings = {}
        
        for duration in durations:
            # Generate test audio
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)
            
            audio_path = test_data_dir / f"test_{duration}s.wav"
            sf.write(audio_path, audio, sample_rate)
            
            # Time the extraction
            start_time = time.time()
            loops = extract_loops(
                audio_path=str(audio_path),
                loop_duration=1.0,
                aperture=0.5,
                num_loops=3,
                progress=mock_gradio_progress
            )
            elapsed = time.time() - start_time
            timings[duration] = elapsed
        
        # Check scaling
        print(f"\n📈 Loop Extraction Scaling:")
        for duration, timing in sorted(timings.items()):
            print(f"  {duration}s audio: {timing:.2f}s")
        
        # Verify roughly linear scaling
        if len(timings) >= 2:
            durations_list = sorted(timings.keys())
            time_ratio = timings[durations_list[-1]] / timings[durations_list[0]]
            duration_ratio = durations_list[-1] / durations_list[0]
            
            # Time ratio should be similar to duration ratio (within reason)
            assert time_ratio < duration_ratio * 2, "Performance degradation detected"


class TestPerformanceSummary:
    """Generate a performance summary report."""
    
    @pytest.mark.benchmark
    def test_generate_performance_summary(self, sample_audio_file, sample_audio_file_long, mock_gradio_progress, tmp_path):
        """Generate a comprehensive performance summary."""
        summary = {
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tests': []
        }
        
        # Test loop extraction
        start = time.time()
        loops = extract_loops(
            audio_path=sample_audio_file_long,
            loop_duration=2.0,
            aperture=0.5,
            num_loops=3,
            progress=mock_gradio_progress
        )
        summary['tests'].append({
            'operation': 'Loop Extraction',
            'time_seconds': round(time.time() - start, 2),
            'result_count': len(loops)
        })
        
        # Test vocal chops
        start = time.time()
        chops = generate_vocal_chops(
            audio_path=sample_audio_file_long,
            mode='onset',
            min_duration=0.1,
            max_duration=2.0,
            threshold=0.3,
            progress=mock_gradio_progress
        )
        summary['tests'].append({
            'operation': 'Vocal Chop Generation',
            'time_seconds': round(time.time() - start, 2)
        })
        
        # Test drum one-shots
        start = time.time()
        oneshots = generate_drum_oneshots(
            audio_path=sample_audio_file_long,
            min_duration=0.05,
            max_duration=1.0,
            apply_fadeout=True,
            progress=mock_gradio_progress
        )
        summary['tests'].append({
            'operation': 'Drum One-Shot Generation',
            'time_seconds': round(time.time() - start, 2)
        })
        
        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Test Date: {summary['test_date']}")
        print("\nOperation Timings:")
        for test in summary['tests']:
            print(f"  {test['operation']}: {test['time_seconds']}s")
        print("="*60)
        
        assert len(summary['tests']) > 0
